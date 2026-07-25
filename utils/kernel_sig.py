"""Extract kernel signatures from reference LLVM IR or CUDA source.

Parses the `define void @mangled_name(params)` line to get:
  - kernel name (mangled)
  - argument list: (type, size_bytes) per arg
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass
class KernelParam:
    ordinal: int
    ctype: str        # C type: "float*", "int", "float", etc.
    llvm_type: str    # LLVM type: "ptr", "i32", "float"
    size: int         # bytes: 8 for ptr, 4 for i32/float
    is_pointer: bool
    is_output: bool   # heuristic: writeonly or last ptr before scalars
    is_byval_struct: bool = False   # BUG-051: byval(%struct.NAME) in LLVM IR
    struct_name: str = ""           # BUG-051: e.g. "float3", "int3", "Params"
    elf_offset: int = -1            # authoritative c[0] byte offset from
                                    # EIATTR_KPARAM_INFO (-1 = unknown)


@dataclass
class KernelSignature:
    name: str         # mangled name
    params: List[KernelParam]
    block_dim: int = None   # launch blockDim.x from EIATTR_MAX_THREADS
                            # (apply_elf_arg_info); None = ELF didn't say.
                            # cudnn bn_bw_1C11_kernel_new<...,512,...> hangs
                            # under the default 256: its intra-CTA reduction
                            # waits on warps 8-15 that were never launched.

    @property
    def demangled_hint(self):
        """Best-effort short name from mangled C++ name."""
        m = re.search(r'_Z\d+(\w+)', self.name)
        return m.group(1) if m else self.name


def _demangle_param_types(mangled: str) -> List[str]:
    """Extract C types from Itanium-mangled function name.

    Handles substitutions (S_, S0_, S1_, ...) per Itanium ABI.
    E.g., _Z15accuracy_kerneliiiPKfPKiPi -> [int,int,int,const float*,const int*,int*]
    E.g., _Z13filter_sharedPiS_PKii      -> [int*,int*,const int*,int]

    Substitutable entities: non-builtin types (pointers, qualified types).
    Builtin types (i, f, d, etc.) are NOT added to the substitution table.
    """
    base_map = {'f': 'float', 'i': 'int', 'd': 'double', 'j': 'unsigned int',
                'l': 'long', 'm': 'unsigned long', 'x': 'long long',
                'y': 'unsigned long long', 'b': 'bool', 'c': 'char',
                'h': 'unsigned char', 's': 'short', 't': 'unsigned short',
                'v': 'void', 'a': 'signed char'}

    # Parse Itanium mangling: _Z [L] <len> <name> <type-chars...>
    m = re.match(r'_ZL?(\d+)', mangled)
    if not m:
        return []
    name_len = int(m.group(1))
    sig_start = m.end() + name_len
    if sig_start >= len(mangled):
        return []
    sig = mangled[sig_start:]

    # Handle template functions: _Z<len><name>I<template-args>E<return-type><params>
    # Template args are between I...E.  For template functions, the return type
    # is encoded after E and before the parameter types.
    template_args = []  # list of C types for T_, T0_, T1_, etc.
    is_template = False
    # Substitution candidates, in first-appearance order (Itanium: S_ = seq#0,
    # S0_ = seq#1, Sn_ = seq#n+1). For a function template, the unscoped
    # template NAME itself is candidate seq#0 — created before the template
    # args are parsed — so `I6__halfS0_E` means args (__half, S0_=__half).
    early_subs = []
    if sig.startswith('I'):
        is_template = True
        early_subs.append('<template-name>')
        # Parse template arguments between I and E
        j = 1
        while j < len(sig) and sig[j] != 'E':
            c = sig[j]
            tb = base_map.get(c)
            if tb:
                template_args.append(tb)
                j += 1
            elif c.isdigit():
                # Length-prefixed struct/class name (substitutable)
                num_start = j
                while j < len(sig) and sig[j].isdigit():
                    j += 1
                tname_len = int(sig[num_start:j])
                tname = sig[j:j + tname_len]
                template_args.append(tname)
                early_subs.append(tname)
                j += tname_len
            elif c == 'S':
                # Substitution reference inside the template args (e.g. the
                # second __half in loss_bwd<__half,__half> mangles as S0_).
                k = j + 1
                while k < len(sig) and sig[k] != '_':
                    k += 1
                idx_str = sig[j + 1:k]
                sub_idx = 0 if idx_str == '' else int(idx_str, 36) + 1
                template_args.append(
                    early_subs[sub_idx] if sub_idx < len(early_subs) else 'int')
                j = k + 1
            elif c == 'L':
                # Literal (non-type) template arg: L<type><value>E, e.g.
                # Lb1E = bool true, Li1E = int 1 (jaccard_is<true, float>).
                # It still occupies a template-arg position for T-refs.
                k = j + 1
                if k < len(sig) and sig[k] in base_map:
                    k += 1
                while k < len(sig) and sig[k] != 'E':
                    k += 1
                template_args.append('int')  # placeholder occupying the slot
                j = k + 1
            else:
                # Unknown template arg — skip
                j += 1
        if j < len(sig) and sig[j] == 'E':
            j += 1  # skip 'E'
        sig = sig[j:]
        # For template functions, the return type is encoded first.
        # Skip it: parse one type and discard.
        # Simple approach: skip a single builtin type character for void/int/float.
        if sig and sig[0] in base_map:
            sig = sig[1:]
        # (For non-builtin return types we'd need full parsing — rare for kernels.)

    types = []
    # Substitution table: list of ctype strings.
    # S_ = subs[0], S0_ = subs[1], S1_ = subs[2], etc.
    # Builtin types are NOT substitutable; only qualified/pointer/struct types.
    subs = []
    # For template functions, the template-id (name<args>) occupies the first
    # substitution slot per Itanium ABI.  Template arguments (including builtins
    # when used as template params) also get substitution slots in practice.
    if is_template:
        subs.extend(early_subs)
    i = 0

    def _resolve_sub(pos):
        """Parse S_, S0_, S1_, ... at pos.  Returns (ctype_str, new_pos) or (None, pos)."""
        if pos >= len(sig) or sig[pos] != 'S':
            return None, pos
        # Find the trailing underscore
        j = pos + 1
        while j < len(sig) and sig[j] != '_':
            j += 1
        if j >= len(sig):
            return None, pos
        idx_str = sig[pos + 1:j]
        if idx_str == '':
            sub_idx = 0
        else:
            try:
                sub_idx = int(idx_str, 36) + 1
            except ValueError:
                return None, pos
        new_pos = j + 1
        if sub_idx < len(subs):
            return subs[sub_idx], new_pos
        return None, pos

    def _parse_one_type(pos):
        """Parse one type starting at pos.  Returns (ctype_str, new_pos) or (None, pos)."""
        if pos >= len(sig):
            return None, pos

        # -- Bare substitution reference (no qualifiers before it): S_, S0_, ...
        # Substitution seq-ids are base-36 DIGITS AND UPPERCASE (S0_..S9_,
        # SA_, SB_, ...); the lowercase letters after S are the std::
        # abbreviations (St, Sa, Ss, ...) which are not substitution refs.
        if sig[pos] == 'S' and pos + 1 < len(sig) and \
                (sig[pos + 1] == '_' or sig[pos + 1].isdigit()
                 or sig[pos + 1].isupper()):
            result, new_pos = _resolve_sub(pos)
            if result is not None:
                return result, new_pos

        # -- Collect qualifiers (P = pointer, K = const, R = reference)
        # In Itanium mangling, qualifiers are outermost-first:
        #   PKf = P(K(f)) = pointer to (const float) = "const float*"
        qualifiers = []
        p = pos
        while p < len(sig) and sig[p] in ('P', 'K', 'R', 'V'):
            qualifiers.append(sig[p])
            p += 1

        if p >= len(sig):
            return None, pos

        # -- Parse the base type (after qualifiers)
        inner_ctype = None
        base_is_sub = False

        if sig[p] == 'T' and is_template:
            # Template parameter reference: T_ = template_args[0], T0_ = template_args[1], etc.
            # Per Itanium ABI the T-reference ITSELF is a substitution
            # candidate (then each qualifier layer adds more) — without this,
            # later S-refs (loss_bwd's S6_/S3_/S8_) resolve to wrong slots.
            tp = p + 1
            if tp < len(sig) and sig[tp] == '_':
                # T_ = first template arg
                if template_args:
                    inner_ctype = template_args[0]
                else:
                    inner_ctype = 'int'  # fallback
                new_pos = tp + 1
                subs.append(inner_ctype)
            elif tp < len(sig) and sig[tp].isdigit():
                # T0_, T1_, etc.
                idx_start = tp
                while tp < len(sig) and sig[tp].isdigit():
                    tp += 1
                if tp < len(sig) and sig[tp] == '_':
                    t_idx = int(sig[idx_start:tp]) + 1
                    inner_ctype = template_args[t_idx] if t_idx < len(template_args) else 'int'
                    new_pos = tp + 1
                    subs.append(inner_ctype)
                else:
                    return None, pos
            else:
                return None, pos
        elif sig[p] == 'S' and p + 1 < len(sig) and \
                (sig[p + 1] == '_' or sig[p + 1].isdigit()
                 or sig[p + 1].isupper()):
            # Substitution as inner type
            inner_ctype, new_pos = _resolve_sub(p)
            if inner_ctype is None:
                return None, pos
            base_is_sub = True
        elif sig[p].isdigit():
            # Struct/class name (length-prefixed)
            num_start = p
            while p < len(sig) and sig[p].isdigit():
                p += 1
            name_len2 = int(sig[num_start:p])
            if p + name_len2 > len(sig):
                return None, pos
            struct_name = sig[p:p + name_len2]
            new_pos = p + name_len2
            inner_ctype = struct_name
            # Struct names are substitutable
            subs.append(struct_name)
            # Class template-id: <len><name> I <args> E (PK13CustomComplexIdE
            # = const CustomComplex<double>*).  The bare template NAME got a
            # slot above; the full instantiation gets its OWN slot per the
            # Itanium ABI — gpp's S6_ only resolves with both registered.
            # Without this branch the parse aborted at 'I' and silently
            # dropped 6 of solver's 13 params (wild pointers -> SIGSEGV).
            if new_pos < len(sig) and sig[new_pos] == 'I':
                targs = []
                q = new_pos + 1
                while q < len(sig) and sig[q] != 'E':
                    if sig[q] == 'L':
                        # Literal (non-type) arg: L<type><value>E, e.g. Li5E
                        e = sig.find('E', q)
                        if e == -1:
                            return None, pos
                        v = sig[q + 1:e]
                        targs.append(v.lstrip(''.join(base_map)) or v)
                        q = e + 1
                        continue
                    t, q2 = _parse_one_type(q)
                    if t is None or q2 == q:
                        return None, pos
                    targs.append(t)
                    q = q2
                if q >= len(sig):
                    return None, pos
                inner_ctype = f"{struct_name}<{', '.join(targs)}>"
                new_pos = q + 1
                subs.append(inner_ctype)
        elif sig[p] == 'A':
            # Array type: A<n>_<elem> (possibly nested: PA24_A24_f =
            # float(*)[24][24]). Driver generation only needs the ELEMENT
            # type and correct arity, so flatten to the innermost element
            # with array suffixes for display.
            dims = []
            q = p
            while q < len(sig) and sig[q] == 'A':
                q += 1
                d0 = q
                while q < len(sig) and sig[q].isdigit():
                    q += 1
                dims.append(sig[d0:q] or '?')
                if q < len(sig) and sig[q] == '_':
                    q += 1
            elem, new_pos = _parse_one_type(q)
            if elem is None:
                return None, pos
            # Each array layer is a substitution candidate, registered
            # inner-first (PA6_A6_A6_f then PS0_ resolves to float[6][6]).
            layer = elem
            for d in reversed(dims):
                layer = f"{layer}[{d}]"
                subs.append(layer)
            inner_ctype = elem + ''.join(f'[{d}]' for d in dims)
        else:
            c = sig[p]
            base = base_map.get(c, None)
            if base is None:
                return None, pos
            inner_ctype = base
            new_pos = p + 1

        if not qualifiers:
            # No qualifiers — return as-is (builtins not registered)
            return inner_ctype, new_pos

        # Apply qualifiers in reverse order (innermost qualifier is rightmost).
        # In mangling "PKf": P is outer, K is inner.
        # Process right-to-left: K first (const float), then P (const float*).
        #
        # Per Itanium ABI substitution rules:
        # - Bare builtin types (float, int, etc.) are NOT substitution candidates.
        # - cv-qualified types (const float), pointer types (float*), and reference
        #   types (&) ARE substitution candidates, even when the inner type is builtin.
        # - Each intermediate qualified type in a compound type is registered.
        #   E.g., PKf -> "const float" [S_] and "const float*" [S0_].
        ctype = inner_ctype
        for q in reversed(qualifiers):
            if q == 'K':
                ctype = f"const {ctype}"
            elif q == 'P':
                ctype = f"{ctype}*"
            elif q == 'R':
                ctype = f"{ctype}&"
            elif q == 'V':
                ctype = f"volatile {ctype}"
            # Every qualified type (including cv-qualified builtins) is substitutable
            subs.append(ctype)

        return ctype, new_pos

    while i < len(sig):
        result = _parse_one_type(i)
        if result[0] is None:
            # Incomplete parse: better no answer than a silently truncated
            # param list (gpp's solver lost 6 of 13 params and the generated
            # driver passed wild pointers).  Caller falls back to
            # llvm-cxxfilt/c++filt.
            return []
        ctype, new_i = result
        types.append(ctype)
        i = new_i

    # `(v)` encodes an EMPTY parameter list.
    if types == ['void']:
        return []
    return types


_tool_demangle_cache = {}


def _demangle_via_tool(mangled: str) -> Optional[str]:
    """Demangle with llvm-cxxfilt / c++filt (battle-tested for the nested
    _ZN names the hand demangler does not cover). Returns None on failure."""
    if mangled in _tool_demangle_cache:
        return _tool_demangle_cache[mangled]
    import shutil as _shutil
    import subprocess as _sp
    result = None
    for tool in ('llvm-cxxfilt', 'c++filt'):
        path = _shutil.which(tool)
        if not path:
            continue
        try:
            out = _sp.run([path], input=mangled + '\n', capture_output=True,
                          text=True, timeout=10).stdout.strip()
        except Exception:
            continue
        if out and out != mangled:
            result = out
            break
    _tool_demangle_cache[mangled] = result
    return result


def _params_from_demangled(text: str) -> List[str]:
    """Split the parameter list of demangled C++ text into type strings.

    Takes the LAST balanced top-level (...) group (the function's params;
    earlier parens belong to template args / NTTP casts) and splits on
    commas at zero <>/()/[] depth.
    """
    end = text.rfind(')')
    if end == -1:
        return []
    depth = 0
    start = -1
    for i in range(end, -1, -1):
        if text[i] == ')':
            depth += 1
        elif text[i] == '(':
            depth -= 1
            if depth == 0:
                start = i
                break
    if start < 0:
        return []
    inner = text[start + 1:end].strip()
    if not inner or inner == 'void':
        return []
    parts, buf, d = [], [], 0
    for ch in inner:
        if ch in '<([':
            d += 1
        elif ch in '>)]':
            d -= 1
        if ch == ',' and d == 0:
            parts.append(''.join(buf).strip())
            buf = []
        else:
            buf.append(ch)
    if buf:
        parts.append(''.join(buf).strip())
    return [p for p in parts if p]


def signature_from_mangled(mangled_name: str) -> Optional[KernelSignature]:
    """Build a KernelSignature directly from an Itanium-mangled kernel name.

    Used when no reference LLVM IR is available (the lifted symbol IS the
    mangled CUDA name, which encodes the full C++ parameter list). Returns
    None for non-mangled names or when demangling yields nothing.
    """
    if not mangled_name.startswith('_Z'):
        return None
    try:
        ctypes = _demangle_param_types(mangled_name)
    except Exception:
        return None
    if not ctypes:
        # Hand demangler failed or bailed mid-parse (namespaced cutlass::…
        # kernels, exotic manglings).  llvm-cxxfilt/c++filt are
        # authoritative; ELF KPARAM info then corrects sizes/offsets.
        demangled = _demangle_via_tool(mangled_name)
        if demangled:
            ctypes = _params_from_demangled(demangled)
    if not ctypes:
        return KernelSignature(name=mangled_name, params=[])
    builtins = {'float', 'int', 'double', 'unsigned int', 'long',
                'unsigned long', 'long long', 'unsigned long long', 'bool',
                'char', 'unsigned char', 'short', 'unsigned short', 'void',
                'signed char', 'int64_t', 'uint64_t', 'size_t'}
    params = []
    last_ptr_idx = -1
    for i, ct in enumerate(ctypes):
        is_ptr = ct.endswith('*')
        if is_ptr:
            last_ptr_idx = i
        base = ct.rstrip('*').replace('const ', '').replace('volatile ', '').strip()
        size = 8 if (is_ptr or base in ('double', 'long', 'int64_t',
                                        'unsigned long', 'long long',
                                        'unsigned long long', 'size_t')) else 4
        # A non-pointer, non-reference, non-builtin token is a by-value
        # struct (or enum — same 4-byte default until the ELF says
        # otherwise).  Size stays at the old guess here; apply_elf_arg_info
        # overwrites it with the authoritative EIATTR_KPARAM_INFO size, which
        # is what actually fixes the layout (PayoffCall: 8B struct guessed as
        # a 4B int shifted every later param -> SIGSEGV, aop-cuda).
        is_struct = (not is_ptr and not ct.endswith('&')
                     and base not in builtins)
        params.append(KernelParam(
            ordinal=i, ctype=ct, llvm_type='ptr' if is_ptr else base,
            size=size, is_pointer=is_ptr,
            is_output=False,
            is_byval_struct=is_struct,
            struct_name=base if is_struct else "",
        ))
    # Heuristic: the output is the last NON-const pointer param.  Const
    # pointers are inputs by definition — falling back to a const pointer
    # (old behavior) made the driver compare an input buffer against a
    # zeroed ref shadow, guaranteed wrong_result (nlll-cuda: signature ends
    # with `const weights*`, real outputs are params 0/1).  Only if every
    # pointer is const do we keep the last one as a best-effort output.
    last_nonconst_ptr_idx = -1
    for p in params:
        if p.is_pointer and 'const' not in p.ctype:
            last_nonconst_ptr_idx = p.ordinal
    out_idx = last_nonconst_ptr_idx if last_nonconst_ptr_idx >= 0 else last_ptr_idx
    if out_idx >= 0:
        params[out_idx].is_output = True
    return KernelSignature(name=mangled_name, params=params)


def _extract_balanced_params(text: str, start: int) -> Optional[str]:
    """Extract parameter list with balanced parens starting at '(' position.

    BUG-051: The simple [^)]* regex fails when params contain byval(%struct.NAME).
    This function properly handles nested parentheses.
    """
    if start >= len(text) or text[start] != '(':
        return None
    depth = 0
    i = start
    while i < len(text):
        if text[i] == '(':
            depth += 1
        elif text[i] == ')':
            depth -= 1
            if depth == 0:
                return text[start + 1:i]  # contents between outer parens
        i += 1
    return None


def extract_from_ll(ll_path: str) -> List[KernelSignature]:
    """Extract kernel signatures from a reference LLVM IR file."""
    text = Path(ll_path).read_text()
    sigs = []

    # BUG-051: Use a two-step approach to handle byval(%struct.NAME) with nested parens.
    # Step 1: find 'define ... void @name(' with a simple regex.
    # Step 2: extract balanced param list from the '(' position.
    for m in re.finditer(
        r'define\s+(?:dso_local\s+)?void\s+@"?([^"(\s]+)"?\(',
        text
    ):
        name = m.group(1)
        paren_pos = m.end() - 1  # position of the opening '('
        param_str = _extract_balanced_params(text, paren_pos)
        if param_str is None:
            continue
        if not param_str.strip():
            sigs.append(KernelSignature(name=name, params=[]))
            continue

        raw_args = _split_params(param_str)
        params = []
        last_ptr_idx = -1
        for i, arg in enumerate(raw_args):
            p = _parse_param(i, arg)
            if p.is_pointer:
                last_ptr_idx = i
            params.append(p)

        # Heuristic: mark output pointers
        # readonly -> input, writeonly -> output, otherwise last ptr is output
        for p in params:
            if p.is_pointer and not p.is_output:
                if p.ordinal == last_ptr_idx:
                    p.is_output = True

        # Refine types from demangled name.
        # BUG-051: When struct by-value params exist, the demangled types include
        # the struct as one type, but in LLVM IR it appears as ptr byval(%struct.NAME).
        # Match accounting for this: demangled count may differ from param count.
        demangled_types = _demangle_param_types(name)
        _apply_demangled_types(params, demangled_types)

        sigs.append(KernelSignature(name=name, params=params))

    return sigs


def _apply_demangled_types(params: List[KernelParam],
                           demangled_types: List[str]) -> None:
    """Apply demangled C types to parsed LLVM IR params.

    BUG-051: Handles the case where demangled types and IR params have different
    counts due to struct by-value parameters.  In the demangled name, a struct
    by-value shows as e.g. 'float3', while in LLVM IR it appears as
    'ptr byval(%struct.float3)'.  So the counts should match 1:1.

    Also handles pointer element type refinement (e.g. 'float*' -> 'const float4*').
    """
    if len(demangled_types) == len(params):
        # Direct 1:1 mapping
        for p, dt in zip(params, demangled_types):
            if p.is_byval_struct and '*' not in dt:
                # by-value struct: set the struct name from demangled type
                p.ctype = dt
                p.struct_name = dt
            elif p.is_pointer and '*' in dt:
                p.ctype = dt
            elif not p.is_pointer and '*' not in dt:
                # scalar — update ctype if demangled gives more info
                # (e.g. 'bool' vs 'int' for 'b' in mangled name)
                if dt == 'bool':
                    p.ctype = 'bool'
                    p.size = 1
    elif len(demangled_types) > len(params):
        # More demangled types than params — should not normally happen after
        # fixing the regex.  Try best-effort alignment.
        pass
    # If fewer demangled types, skip refinement (already defaulted).


def _split_params(param_str: str) -> List[str]:
    """Split param string respecting nested parens."""
    parts = []
    depth = 0
    current = []
    for ch in param_str:
        if ch == ',' and depth == 0:
            parts.append(''.join(current).strip())
            current = []
        else:
            if ch == '(':
                depth += 1
            elif ch == ')':
                depth -= 1
            current.append(ch)
    if current:
        parts.append(''.join(current).strip())
    return [p for p in parts if p]


def _parse_param(ordinal: int, arg_text: str) -> KernelParam:
    """Parse a single LLVM IR parameter declaration."""
    # BUG-051: Detect byval(%struct.NAME) before tokenizing
    byval_match = re.search(r'byval\(%struct\.(\w+)\)', arg_text)
    is_byval = byval_match is not None
    struct_name = byval_match.group(1) if byval_match else ""

    tokens = arg_text.split()
    # Remove parameter name (%N)
    tokens = [t for t in tokens if not t.startswith('%')]
    # Remove attributes
    attrs = {'noalias', 'nocapture', 'noundef', 'readonly', 'writeonly',
             'nonnull', 'dereferenceable', 'align', 'inreg', 'signext', 'zeroext',
             'local_unnamed_addr', 'dso_local'}

    is_readonly = 'readonly' in tokens
    is_writeonly = 'writeonly' in tokens

    clean = [t for t in tokens if t not in attrs and not t.startswith('dereferenceable')
             and not t.startswith('byval(')]

    # BUG-051: byval struct params appear as 'ptr ... byval(%struct.NAME)' in LLVM IR.
    # These are NOT regular pointer params — they are struct by-value params that CUDA
    # serializes into const_mem as individual scalar fields.
    if is_byval and 'ptr' in clean:
        struct_size = _struct_size(struct_name)
        return KernelParam(
            ordinal=ordinal, ctype=struct_name, llvm_type="ptr",
            size=struct_size, is_pointer=False, is_output=False,
            is_byval_struct=True, struct_name=struct_name,
        )

    if 'ptr' in clean:
        # Pointer — default to float*, refined by demangle_params() later
        return KernelParam(
            ordinal=ordinal, ctype="float*", llvm_type="ptr",
            size=8, is_pointer=True, is_output=is_writeonly,
        )
    elif 'i32' in clean:
        return KernelParam(
            ordinal=ordinal, ctype="int", llvm_type="i32",
            size=4, is_pointer=False, is_output=False,
        )
    elif 'i64' in clean:
        return KernelParam(
            ordinal=ordinal, ctype="int64_t", llvm_type="i64",
            size=8, is_pointer=False, is_output=False,
        )
    elif 'float' in clean:
        return KernelParam(
            ordinal=ordinal, ctype="float", llvm_type="float",
            size=4, is_pointer=False, is_output=False,
        )
    elif 'double' in clean:
        return KernelParam(
            ordinal=ordinal, ctype="double", llvm_type="double",
            size=8, is_pointer=False, is_output=False,
        )
    else:
        # Unknown — treat as i32
        return KernelParam(
            ordinal=ordinal, ctype="int", llvm_type="i32",
            size=4, is_pointer=False, is_output=False,
        )


# BUG-051: Known CUDA vector/struct types and their sizes (bytes).
# Used for byval struct decomposition into scalar fields.
_KNOWN_STRUCTS = {
    # CUDA vector types
    'float2': [('float', 4), ('float', 4)],             # 8 bytes
    'float3': [('float', 4), ('float', 4), ('float', 4)],  # 12 bytes
    'float4': [('float', 4), ('float', 4), ('float', 4), ('float', 4)],  # 16 bytes
    'int2':   [('int', 4), ('int', 4)],                  # 8 bytes
    'int3':   [('int', 4), ('int', 4), ('int', 4)],      # 12 bytes
    'int4':   [('int', 4), ('int', 4), ('int', 4), ('int', 4)],  # 16 bytes
    'double2': [('double', 8), ('double', 8)],            # 16 bytes
    'uint2':  [('int', 4), ('int', 4)],                   # 8 bytes
    'uint3':  [('int', 4), ('int', 4), ('int', 4)],       # 12 bytes
    'uint4':  [('int', 4), ('int', 4), ('int', 4), ('int', 4)],  # 16 bytes
    'char2':  [('int', 4), ('int', 4)],                   # promote to 4-byte for const_mem alignment
    'char3':  [('int', 4), ('int', 4), ('int', 4)],
    'char4':  [('int', 4), ('int', 4), ('int', 4), ('int', 4)],
    'short2': [('int', 4), ('int', 4)],
    'short3': [('int', 4), ('int', 4), ('int', 4)],
    'short4': [('int', 4), ('int', 4), ('int', 4), ('int', 4)],
    'uchar2': [('int', 4), ('int', 4)],
    'uchar3': [('int', 4), ('int', 4), ('int', 4)],
    'uchar4': [('int', 4), ('int', 4), ('int', 4), ('int', 4)],
    'ushort2': [('int', 4), ('int', 4)],
    'ushort3': [('int', 4), ('int', 4), ('int', 4)],
    'ushort4': [('int', 4), ('int', 4), ('int', 4), ('int', 4)],
    'longlong2': [('int64_t', 8), ('int64_t', 8)],
    'ulonglong2': [('int64_t', 8), ('int64_t', 8)],
    # cudnnTensorStruct (by-value tensor descriptor on every cudnn_ops bn/
    # pooling kernel): dataType/format/nbDims + dimA[8] + strideA[8] + tail
    # words — ALL i32, no pointers, no i64s. 104 bytes matches the
    # EIATTR_KPARAM_INFO size on every sm_90a cudnn_ops cubin (windows
    # 0x210..0x278 / 0x280..0x2e8). Registering it disables the window
    # slot scans for these params: use-pattern inference misread the
    # descriptor copy idiom — vectorized ULDC.64 over adjacent dims
    # classified as i64 scalars zeroed the hi dim (bn_fw_tr_1C11 NCHW
    # count = N*0 = 0 -> NaN), and the b1=0 variants' STL spill of the
    # struct produced 21 such misfills -> SIGSEGV. Type knowledge beats
    # inference; every word is a small-int N fill.
    'cudnnTensorStruct': [('int', 4)] * 26,             # 104 bytes
}


def _struct_size(name: str) -> int:
    """Get byte size of a known struct, or estimate from name."""
    if name in _KNOWN_STRUCTS:
        return sum(sz for _, sz in _KNOWN_STRUCTS[name])
    # Heuristic: default to 16 bytes for unknown structs
    return 16


# Per-field fill overrides for known structs: field index -> C literal.
# The generic decomposition fills every int field with N (=32); fields with
# STRUCTURAL meaning need domain-valid values instead. cudnnTensorStruct
# word 2 is nbDims (the SASS guards it with ISETP.GT 0x4 for the 4D/5D
# split): N=32 made the nbDims-bounded dims*strides product walk all 16
# dimA/strideA words, overflowing the element count to ~2^80 — every
# bn_fw_tr_1C11_kernel_NCHW<..,false,..,..> variant indexed x[] gigabytes
# out of the map (24 deterministic SIGSEGVs on so.75). Real callers pass
# nbDims=4/5; 4 bounds the product at 32^3*32 and matches the dims the
# kernels actually read.
#
# Words 16..23 are strideA[8] (+0x40..; dimA[8] is words 8..15 at +0x20..,
# both per the recovered SASS field maps). The uniform N
# fill gave every stride 32 while the innermost row is dims[2]*dims[3] =
# 1024 elements long, so tensor offsets n*strideA[0] + c*strideA[1] + j
# COLLIDED across batch rows (off(0,0,32) == off(1,0,0)): the bn beta-blend
# read-modify-writes (y = a*(..) + beta*y_old) became order-dependent under
# the concurrent wrapper threads — a value-racy config the smoke check
# cannot see. Strides must satisfy
# stride[i] >= product of inner dims; with every dim filled N and nbDims=4
# the packed-contiguous choice is {N^3, N^2, N, 1} (exactly what real NCHW
# callers pass). strideA[4..7] are beyond nbDims=4 — never read — and get
# the innermost stride 1. Max touched offset (dims[i]-1)*stride[i] summed =
# N^4 - 1 = 2^20-1 << BUF_SIZE 16M. Dims words keep the N fill (the
# non-vacuous M=N*N^2 element count is preserved).
_KNOWN_STRUCT_FILLS = {
    'cudnnTensorStruct': {
        2: '4',                                    # nbDims
        16: 'N*N*N', 17: 'N*N', 18: 'N', 19: '1',  # strideA[0..3] packed
        20: '1', 21: '1', 22: '1', 23: '1',        # strideA[4..7] (> nbDims)
    },
}

# Driver-text marker for the CURRENT known-struct fill semantics. host_gen
# emits it whenever a decomposed by-value param has fill overrides;
# run_hecbench regenerates cached drivers that decompose such a struct but
# lack the tag (same mechanism as the "(pair-aware, icmp-dim)" i64-scan
# marker). Bump the version when _KNOWN_STRUCT_FILLS semantics change.
KNOWN_STRUCT_FILL_TAG = 'known-struct fills v2 (dims-consistent strides)'


def get_struct_fields(name: str) -> List[tuple]:
    """Get (ctype, size_bytes) list for a struct's fields.

    Returns empty list for unknown structs (caller should use default handling).
    """
    return _KNOWN_STRUCTS.get(name, [])


def get_struct_fill_overrides(name: str) -> dict:
    """Per-field fill literals for a known struct ({field_index: literal}).

    Empty for structs whose generic N/2.0 fills are domain-valid.
    """
    return _KNOWN_STRUCT_FILLS.get(name, {})


# ---------------------------------------------------------------------------
# Authoritative param layout from the ELF (EIATTR_KPARAM_INFO)
# ---------------------------------------------------------------------------

_kparam_cache = {}


def _elf_arg_map(elf_path):
    """kernel -> ([(abs_offset, size)] per ordinal, max_threads) from a
    SASS ELF: (arg_map, max_threads) tuple.

    Elf() handles both cuobjdump -elf text dumps (hecbench *_sm90.elf) and
    binary cubins (cudnn/flash/cutlass). Cached per path.
    """
    key = str(elf_path)
    if key in _kparam_cache:
        return _kparam_cache[key]
    from utils.elf import Elf
    try:
        e = Elf(Path(elf_path))
        info = (e.arg_map, e.max_threads)
    except Exception:
        info = ({}, {})
    _kparam_cache[key] = info
    return info


def apply_elf_arg_info(sig: KernelSignature, elf_path) -> bool:
    """Overwrite guessed param sizes with EIATTR_KPARAM_INFO truth and record
    each param's absolute c[0] offset (KernelParam.elf_offset).

    host_gen emits an ArgAt{offset} pin before every param that has an
    elf_offset, making the driver layout exact regardless of how wrong the
    demangled-type size guesses were. Returns True if applied.
    """
    if elf_path is None:
        return False
    arg_map, max_threads = _elf_arg_map(elf_path)
    entries = arg_map.get(sig.name)
    mt = max_threads.get(sig.name)
    if entries is None:
        # _ZL (internal linkage) vs _Z mismatch between IR and ELF
        alt = sig.name.replace('_ZL', '_Z', 1) if sig.name.startswith('_ZL') \
            else '_ZL' + sig.name[2:] if sig.name.startswith('_Z') else None
        if alt:
            entries = arg_map.get(alt)
            mt = max_threads.get(alt)
    if mt and mt[0] > 0:
        # EIATTR_MAX_THREADS: the compiled-in launch block size. Drivers
        # that launch fewer threads hang kernels whose intra-CTA logic
        # expects the full complement (bn_bw_1C11_kernel_new BD=512).
        sig.block_dim = int(mt[0])
    if not entries or any(e is None for e in entries):
        return False
    if len(entries) != len(sig.params):
        # Demangler returns 0 params for deeply templated names (e.g. CUTLASS
        # kernel_dump). If we have ELF truth and no params yet, synthesize
        # opaque byval-struct params from the ELF offsets/sizes so
        # detect_param_ptr_slots can do its window scan.
        if sig.params:
            return False   # partial mismatch — don't guess
        for i, (off, sz) in enumerate(entries):
            sig.params.append(KernelParam(
                ordinal=i, ctype=f"__elf_param_{i}",
                llvm_type="ptr", size=sz,
                is_pointer=False, is_output=False,
                is_byval_struct=True,
                struct_name=f"__elf_param_{i}",
                elf_offset=off,
            ))
        return True
    for p, (off, sz) in zip(sig.params, entries):
        p.elf_offset = off
        if not p.is_pointer:
            p.size = sz
    return True
