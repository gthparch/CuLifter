"""Match a HeCbench reference.h CPU function to a lifted kernel signature.

Each test/hecbench/<bench>-cuda/ may ship a `reference.h` containing one
or more CPU implementations of the kernel(s) in that benchmark. Names
follow no single convention (we see `*_reference`, `*_ref`, `*_cpu`,
`*Kernel_cpu`, plain `reference`), and signatures may differ from the
kernel's signature (different arg order, additional dim parameters,
templated types).

This module:
  1. Parses reference.h into a list of `RefFunc` records.
  2. Given a `KernelSignature`, picks the best matching `RefFunc` —
     same number of pointer args, same number of scalar args, types
     that are at least loosely compatible, name that looks reference-y.
  3. Returns either a `RefMatch` describing how the driver should call
     the reference (which kernel-side variable name maps to which
     reference arg) or None if no good match exists.

The matcher is intentionally conservative: when it can't be sure the
match is right, it returns None and the driver generator falls back to
smoke validation. Better to keep the smoke pass than to invent a
"correct" answer with a wrong reference call.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

from utils.kernel_sig import KernelSignature, KernelParam


# Tokens that mark a function as a reference / CPU implementation.
# Order matters — earlier entries are preferred when multiple functions
# in one reference.h could match.
_REF_NAME_HINTS = (
    "_reference",
    "_ref",
    "Kernel_cpu",
    "_cpu",
    "kernel_cpu",
    "cpu_",
    "reference",
)


@dataclass
class RefArg:
    """One argument of a reference-function declaration."""

    raw: str          # full argument source as it appears in reference.h
    ctype: str        # cleaned C type, e.g. "const float*", "int"
    is_pointer: bool  # True if ctype ends in '*' (incl. arrays)
    is_const: bool    # True if 'const' precedes the type (input-only)


@dataclass
class RefFunc:
    """A function declared in reference.h."""

    name: str
    template_params: List[str]   # ['typename T', 'int N'] etc., empty if not templated
    args: List[RefArg]
    source: str = ""             # Full function source (signature + body) so the
                                  # driver can embed it inline rather than including
                                  # the whole reference.h, which often pulls in
                                  # unrelated CUDA-only types (half, __nv_bfloat16, …).


@dataclass
class RefMatch:
    """How the driver should call a reference function.

    `arg_exprs` is a list of C++ expressions (one per reference-function
    argument) that the driver can drop into the call site. For inputs
    these are the kernel's own buffer/scalar variables; for outputs they
    are freshly-allocated `r_buf*` shadow buffers.
    """

    func: RefFunc
    arg_exprs: List[str]
    template_args: List[str]            # explicit instantiation, e.g. ['float']
    shadow_pairs: List[Tuple[str, str]] # [(kernel_var, reference-shadow_var)]
    output_pairs: List[Tuple[str, str]] # [(kernel_var, ref_var)] for each output
    elem_type: str                      # element type of the primary output buffer (for tol)
    header_source: str = ""             # Full reference.h (comment-stripped) — embedded
                                         # in the driver so helper functions called by the
                                         # matched function are available too.


# ---------------------------------------------------------------------------
# Reference-header parsing
# ---------------------------------------------------------------------------


def _strip_comments(src: str) -> str:
    src = re.sub(r"/\*.*?\*/", "", src, flags=re.DOTALL)
    src = re.sub(r"//[^\n]*", "", src)
    return src


def _split_args(arg_blob: str) -> List[str]:
    """Split a C++ argument list on top-level commas (template-aware)."""
    out: List[str] = []
    depth = 0
    cur = []
    for ch in arg_blob:
        if ch in "<([{":
            depth += 1
        elif ch in ">)]}":
            depth -= 1
        if ch == "," and depth == 0:
            out.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    tail = "".join(cur).strip()
    if tail:
        out.append(tail)
    return out


def _parse_arg(raw: str) -> RefArg:
    """Parse one C++ argument into a RefArg.

    Best-effort: handles `const T*`, `T*`, `T&`, `T name[]`, `__restrict__`,
    pointer-to-pointer (rare). Drops the parameter name if present.
    """
    s = raw.strip()
    # Drop trailing "= default" if any.
    s = re.sub(r"=\s*[^,]+$", "", s).strip()
    # Drop __restrict__ qualifier — keep type info intact.
    s = re.sub(r"__restrict__\b", "", s).strip()
    # Drop bracketed array dim ("[]" or "[N]") and treat as pointer.
    array_form = bool(re.search(r"\[[^\]]*\]\s*$", s))
    s = re.sub(r"\[[^\]]*\]\s*$", "", s).strip()
    # Last whitespace-separated token is the parameter name (or part of type
    # if anonymous). Heuristic: if the last token contains '*', '&', or is
    # a known type keyword, the arg is anonymous.
    parts = s.split()
    if parts and re.match(r"^[A-Za-z_]\w*$", parts[-1]) and parts[-1] not in (
        "int", "float", "double", "char", "short", "long", "bool",
        "void", "size_t", "uint8_t", "uint16_t", "uint32_t", "uint64_t",
        "int8_t", "int16_t", "int32_t", "int64_t", "half", "__half",
        "bfloat16", "__nv_bfloat16",
    ):
        # Strip the parameter name.
        ctype = " ".join(parts[:-1]).strip()
    else:
        ctype = s.strip()

    is_const = bool(re.match(r"^\s*const\b", ctype))
    is_pointer = ctype.endswith("*") or array_form
    if array_form and not ctype.endswith("*"):
        ctype = ctype + "*"

    return RefArg(raw=raw.strip(), ctype=ctype, is_pointer=is_pointer, is_const=is_const)


_TEMPLATE_RE = re.compile(
    r"template\s*<\s*([^>]*)\s*>\s*(?:inline\s+)?(?:static\s+)?(?:constexpr\s+)?"
    r"([A-Za-z_][\w:<>,\s]*?)\s+([A-Za-z_]\w*)\s*\(([^)]*)\)\s*\{",
    flags=re.DOTALL,
)
_PLAIN_RE = re.compile(
    r"(?:^|\n)\s*(?:inline\s+)?(?:static\s+)?(?:constexpr\s+)?"
    r"(void|[A-Za-z_][\w:<>,\s]*?)\s+([A-Za-z_]\w*)\s*\(([^)]*)\)\s*\{",
    flags=re.DOTALL,
)


def _capture_body(src: str, open_brace_idx: int) -> int:
    """Given the index of `{` that opens a function body, return the
    index just past the matching `}`. Brace-counting only — assumes
    no braces inside string/char literals (true for reference.h).
    """
    depth = 0
    i = open_brace_idx
    n = len(src)
    while i < n:
        ch = src[i]
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    return n


def _extract_decl_start(src: str, body_start: int, name: str) -> int:
    """Walk back from the function-body `{` to the start of the
    declaration (either `template<...>` or the return type). Used to
    capture the full function source for embedding into the driver.
    """
    # Try to find a preceding `template<...>` whose `>` ends before body_start.
    head = src[:body_start]
    # Last `template<...>` whose closing `>` precedes the function name.
    template_end = head.rfind(">")
    template_idx = head.rfind("template", 0, template_end + 1) if template_end != -1 else -1
    name_idx = head.rfind(name, 0, body_start)
    # If template appears between the previous function and our name, use it.
    if template_idx != -1 and name_idx != -1 and template_idx < name_idx:
        # Make sure no other `}` separates them (i.e., the template applies to us).
        between = head[template_idx:name_idx]
        if "}" not in between:
            return template_idx
    # Otherwise back up to the start of this declaration: previous `}` or
    # `;` or beginning-of-file, whichever comes last before name_idx.
    if name_idx == -1:
        return body_start
    cut = max(head.rfind("}", 0, name_idx), head.rfind(";", 0, name_idx))
    return cut + 1 if cut != -1 else 0


def parse_reference_h(path: Path) -> List[RefFunc]:
    """Extract function declarations from a reference.h file.

    Only reports functions returning void or a numeric type. Skips
    leftover artefacts like `arg_max` helpers used internally; the
    matcher will further filter by signature shape.
    """
    if not path.exists():
        return []
    src = _strip_comments(path.read_text(errors="replace"))
    funcs: List[RefFunc] = []

    # Templated forms first (their `template<...>` line precedes the func).
    for m in _TEMPLATE_RE.finditer(src):
        tparams_blob, ret, name, args_blob = m.groups()
        tparams = [t.strip() for t in _split_args(tparams_blob) if t.strip()]
        if not tparams:
            continue
        args = [_parse_arg(a) for a in _split_args(args_blob)]
        body_open = m.end() - 1  # the `{`
        body_close = _capture_body(src, body_open)
        decl_start = _extract_decl_start(src, body_open, name)
        source = src[decl_start:body_close].strip()
        funcs.append(RefFunc(name=name, template_params=tparams, args=args, source=source))

    # Plain (non-templated) forms.
    for m in _PLAIN_RE.finditer(src):
        ret, name, args_blob = m.groups()
        if name in {f.name for f in funcs}:
            # Already captured as a templated form earlier.
            continue
        args = [_parse_arg(a) for a in _split_args(args_blob)]
        body_open = m.end() - 1
        body_close = _capture_body(src, body_open)
        decl_start = _extract_decl_start(src, body_open, name)
        source = src[decl_start:body_close].strip()
        funcs.append(RefFunc(name=name, template_params=[], args=args, source=source))

    return funcs


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------


def _looks_like_reference(name: str) -> int:
    """Score a function name by how 'reference-like' it is. Lower is better."""
    for i, hint in enumerate(_REF_NAME_HINTS):
        if hint in name:
            return i
    return len(_REF_NAME_HINTS) + 100  # not reference-y; deprioritise


def _ctype_compatible(kernel_ct: str, ref_ct: str) -> bool:
    """Loose compatibility: ignore const, normalise pointer levels."""
    def norm(t: str) -> str:
        t = re.sub(r"\bconst\b", "", t)
        t = re.sub(r"\s+", "", t)
        return t

    a = norm(kernel_ct)
    b = norm(ref_ct)
    if a == b:
        return True
    # Templated 'T*' or 'scalar_t*' on the ref side; treat as wildcard.
    if "*" in b and re.search(r"^[A-Za-z_]\w*\*$", b):
        return a.endswith("*")
    return False


_CONCRETE_PTR_TYPES = {
    "int", "float", "double", "char", "short", "long", "bool", "void",
    "size_t", "uint8_t", "uint16_t", "uint32_t", "uint64_t",
    "int8_t", "int16_t", "int32_t", "int64_t", "half", "__half",
    "bfloat16", "__nv_bfloat16", "longlong", "uchar", "ushort", "uint", "ulong",
}


def _cast_if_needed(var: str, ref_ctype: str, kern_ctype: str) -> str:
    """Wrap `var` in a `reinterpret_cast` to `ref_ctype` if needed.

    Skip the cast when types are equivalent (modulo const/whitespace) or
    when the ref type is a bare unbound typename like `T*` (handled at
    template instantiation, not via cast).
    """
    norm = lambda t: re.sub(r"\bconst\b|\s+", "", t)
    rn = norm(ref_ctype)
    if rn == norm(kern_ctype):
        return var
    # Bare typename pointers: ref says `T*` for some unbound T. The
    # template instantiation already binds T to the kernel element type,
    # so no cast is needed here.
    # Test the original spelling (apart from `const`) here. Removing all
    # whitespace first turns concrete multi-token types such as
    # `unsigned char*` into the identifier-looking `unsignedchar*` and
    # incorrectly treats them as template type variables.
    bare = re.sub(r"\bconst\b", "", ref_ctype).strip()
    m = re.fullmatch(r"([A-Za-z_]\w*)\s*\*", bare)
    if m and m.group(1) not in _CONCRETE_PTR_TYPES:
        return var
    return f"reinterpret_cast<{ref_ctype}>({var})"


def _elem_type(ctype: str) -> str:
    """Strip pointer/const qualifiers from a ctype, leaving the element type."""
    t = re.sub(r"\bconst\b", "", ctype).strip()
    if t.endswith("*"):
        t = t[:-1].strip()
    return t


def match_reference(
    sig: KernelSignature,
    ref_funcs: List[RefFunc],
    var_names: List[str],
    header_source: str = "",
    alloc_ctypes: Optional[List[str]] = None,
) -> Optional[RefMatch]:
    """Find a reference function whose signature lines up with the kernel.

    `var_names` is the list of C++ identifier names the driver uses for
    each kernel parameter (`buf0`, `buf1`, ..., `N`, `2.0f`, ...). When
    a reference arg is a pointer that the kernel marks as output, the
    driver allocates a sibling `r_<var>` shadow buffer; otherwise we
    pass the kernel-side variable directly.

    Returns None if no plausible match exists.
    """
    if not ref_funcs:
        return None

    kernel_ptr_count = sum(1 for p in sig.params if p.is_pointer)
    kernel_scalar_count = sum(1 for p in sig.params if not p.is_pointer)

    candidates = []
    for f in ref_funcs:
        ref_ptrs = sum(1 for a in f.args if a.is_pointer)
        ref_scalars = len(f.args) - ref_ptrs

        # Same shape? Require an exact pointer-count match and the same
        # number of scalars (or off-by-one to allow kernels that take a
        # block-dim arg the reference doesn't need).
        if ref_ptrs != kernel_ptr_count:
            continue
        if abs(ref_scalars - kernel_scalar_count) > 1:
            continue
        candidates.append(f)

    if not candidates:
        return None

    # Prefer the most reference-y name.
    candidates.sort(key=lambda f: _looks_like_reference(f.name))
    chosen = candidates[0]
    if _looks_like_reference(chosen.name) >= len(_REF_NAME_HINTS) + 100:
        # No reference-like name at all — skip rather than risk calling
        # a private helper.
        return None

    # Pair up reference args with kernel params.
    # Strategy: walk both arg lists in order; for each kernel param
    # whose `is_pointer` matches the corresponding ref arg's pointer
    # status, use the kernel-side variable name. If shapes differ
    # (extra leading scalar in either side), try a one-arg shift.
    arg_exprs: List[str] = []
    shadow_pairs: List[Tuple[str, str]] = []
    output_pairs: List[Tuple[str, str]] = []
    elem_type = "float"
    # Track typename->kernel-element-type bindings inferred from positional
    # arg pairing. The first kernel pointer that lines up with a ref pointer
    # whose element type is a bare typename (T/U/scalar_t) wins.
    type_bindings: dict = {}

    def _maybe_bind(ref_ct: str, kparam_ct: str):
        """If `ref_ct` is `[const ]T*` for some bare typename T, record the
        binding T -> elem(kparam_ct)."""
        m = re.match(r"^\s*(?:const\s+)?([A-Za-z_]\w*)\s*\*\s*$", ref_ct)
        if not m:
            return
        tname = m.group(1)
        # Skip concrete types — they don't need binding.
        concrete = {"int", "float", "double", "char", "short", "long", "bool",
                    "void", "size_t", "uint8_t", "uint16_t", "uint32_t", "uint64_t",
                    "int8_t", "int16_t", "int32_t", "int64_t", "half", "__half",
                    "bfloat16", "__nv_bfloat16"}
        if tname in concrete:
            return
        kelem = _elem_type(kparam_ct)
        if kelem and tname not in type_bindings:
            type_bindings[tname] = kelem

    # Build kernel ptr/scalar index streams. Pair each (param, var) with
    # the actual driver-side allocation type so the cast helper sees the
    # type the variable really has (e.g. int8_t*, not bool*).
    if alloc_ctypes is None:
        alloc_ctypes = [p.ctype for p in sig.params]
    kernel_iter = iter(zip(sig.params, var_names, alloc_ctypes))
    for ref_arg in chosen.args:
        try:
            kparam, kvar, kalloc = next(kernel_iter)
        except StopIteration:
            return None
        # Skip a kernel scalar that the reference doesn't take.
        while ref_arg.is_pointer != kparam.is_pointer:
            if ref_arg.is_pointer and not kparam.is_pointer:
                # Ref expects pointer, kernel offers scalar — skip the scalar.
                try:
                    kparam, kvar, kalloc = next(kernel_iter)
                except StopIteration:
                    return None
            else:
                # Ref expects scalar, kernel offers pointer at this position
                # — bail rather than scramble the order.
                return None

        if ref_arg.is_pointer:
            _maybe_bind(ref_arg.ctype, kalloc)
            # The reference must see the same pre-launch state as the lifted
            # kernel. Pass a shadow copy for every pointer, not the possibly
            # mutated live buffer. A non-const reference pointer is an
            # authoritative output/in-out marker and is stronger than the
            # mangled-signature last-pointer heuristic.
            r_var = f"r_{kvar}"
            if (kvar, r_var) not in shadow_pairs:
                shadow_pairs.append((kvar, r_var))
        if ref_arg.is_pointer and not ref_arg.is_const:
            output_pairs.append((kvar, r_var))
            if elem_type == "float":
                # Use the kernel param's element type for tolerance choice.
                elem_type = _elem_type(kalloc) or _elem_type(ref_arg.ctype) or "float"
            arg_exprs.append(_cast_if_needed(r_var, ref_arg.ctype, kalloc))
        elif ref_arg.is_pointer:
            arg_exprs.append(_cast_if_needed(r_var, ref_arg.ctype, kalloc))
        else:
            arg_exprs.append(kvar)

    if not output_pairs:
        # The reference touched no output buffer; nothing to compare.
        return None

    # Resolve template instantiation. Prefer a binding inferred from
    # positional pairing (T/U/scalar_t pointer arg matched to a concrete
    # kernel arg); fall back to the primary output buffer's element type.
    template_args: List[str] = []
    for tp in chosen.template_params:
        # 'typename T', 'class U', or 'int N'.
        if "typename" in tp or "class" in tp:
            # Last token of the template-parameter spec is the name (T/U/...).
            tname = tp.strip().split()[-1].rstrip(",")
            bound = type_bindings.get(tname)
            template_args.append(bound or (elem_type if elem_type != "void" else "float"))
        else:
            # Non-type parameter — we can't generally guess. Bail.
            return None

    return RefMatch(
        func=chosen,
        arg_exprs=arg_exprs,
        template_args=template_args,
        shadow_pairs=shadow_pairs,
        output_pairs=output_pairs,
        elem_type=elem_type,
        header_source=header_source,
    )
