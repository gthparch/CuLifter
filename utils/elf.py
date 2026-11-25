from dataclasses import dataclass
from pathlib import Path
import re
import struct


@dataclass(frozen=True)
class KernelArg:
    ordinal: int
    offset: int
    size: int


_ELF_MAGIC = b'\x7fELF'


class Elf:
    """
    Parsed view of a CUDA ELF — either a text dump (cuobjdump -elf) or a raw
    binary cubin.

    This class is intentionally lightweight: it parses enough metadata for SLifter
    to understand kernel parameter layout (.nv.info.*) and constant memory initial
    contents (.nv.constant*.*).
    """

    def __init__(self, path):
        self.path = Path(path)
        self.blackwell_param_abi = False
        self.arg_map = {}
        self.param_cbank_base = {}
        self.const_data = {}
        # Shared constant banks (`.nv.constant<N>` with NO `.<kernel>` suffix):
        # module-wide tables — e.g. nvjpeg's IDCT zigzag in bank 3 — that every
        # kernel reading c[N][...] shares. Keyed by bank -> words; const_words()
        # falls back here when a kernel has no per-kernel bank of its own.
        self.shared_const_data = {}
        self.rel_text_call_map = {}
        # Bank-4 (.rela.nv.constant4) relocations: byte offset in c[0x4] ->
        # (symbol name, symbol value or None if UNDEF, is_function). Slots in
        # bank 4 hold device-function/global addresses the CUDA loader fills;
        # ResolveIndirectCall reads this to turn `LDC.64 Rx, c[0x4][imm]` +
        # `CALL.ABS.NOINC Rx` into a direct call.
        self.const4_relocs = {}
        # Device-global sections. `.nv.global.init` (PROGBITS) carries
        # initialized __device__ globals (e.g. curand's precomputed XORWOW
        # skip-ahead matrices); `.nv.global` (NOBITS) is the zero-init BSS.
        # The CUDA loader allocates both and patches references; the lifter
        # must materialize them or every table read comes back 0/faults.
        self.nv_global_init = b""   # .nv.global.init section bytes
        self.nv_global_size = 0     # .nv.global (NOBITS) size in bytes
        # symbol name -> (in_init_section: bool, offset, size)
        self.nv_global_syms = {}
        # ABS32 immediate relocations against device globals:
        # kernel -> {instruction text offset -> (symbol, is_hi_half)}.
        # SASS materializes a global's address as `UMOV URx, 0x0` pairs the
        # loader patches via R_CUDA_ABS32_LO_32/HI_32 (types 56/57).
        self.rel_text_abs32 = {}
        self.frame_size = {}  # kernel -> stack frame size in bytes
        # kernel -> total stack depth in bytes (EIATTR_MIN_STACK_SIZE). This is
        # the kernel's own frame PLUS the deepest chain of callee frames. When
        # the lifter INLINES device-function calls into one kernel, each former
        # callee keeps its `IADD3 R1, R1, -<callee_frame>` prologue that chains
        # off R1 (the per-thread SP). The driver populates c[0x0][0x28] with the
        # SP base; the lifter materialises that base into c[0x0][0x28] so the
        # chain stays in-bounds. The base must cover the DEEPEST carve, i.e.
        # min_stack_size, not just the kernel's own frame_size (which is 0 when
        # all locals live in inlined callees — e.g. fresnel: frame=0, stack=0x1a0).
        self.min_stack_size = {}  # kernel -> total stack depth in bytes
        # Per-kernel STATIC shared-memory size (bytes) from `.nv.shared.<kernel>`
        # (NOBITS). The NVVM lifter sizes @shared_mem to the largest kernel;
        # the old fixed 32 KB silently overflowed bigger __shared__ (tensorT's
        # double tile[5900] = 47200 B -> cross-thread corruption).
        self.shared_sizes = {}  # kernel -> static shared bytes
        self.guarded_tmem_entries = set()
        # kernel -> (x, y, z) launch block dims from EIATTR_MAX_THREADS. The
        # lifter emits these as nvvm `maxntid*` so ptxas caps register usage to
        # fit the original block size; without them ptxas over-allocates regs
        # and large-block launches (e.g. 1024-thread LZW/resize) fail at launch
        # with "too many resources requested".
        self.max_threads = {}
        self.cluster_dims = {}  # fixed CTA grid per cluster (EIATTR_CTA_PER_CLUSTER)

        # Detect binary ELF vs text dump
        with open(self.path, 'rb') as f:
            magic = f.read(4)

        if magic == _ELF_MAGIC:
            # Binary cubin — parse with pyelftools
            self._text = self.path.read_text(encoding="utf-8", errors="ignore")
            self._parse_binary_cubin()
        else:
            self._text = self.path.read_text(encoding="utf-8", errors="ignore")
            self.blackwell_param_abi = bool(re.search(r"\b(?:EF_CUDA_SM|sm_)(?:100|120)a?\b", self._text, re.I))
            self._parse(self._text)

    @property
    def text(self):
        return self._text

    def kernel_args(self, kernel):
        entries = self.arg_map.get(kernel) or []
        out = []
        for ordinal, entry in enumerate(entries):
            if entry is None:
                continue
            offset, size = entry
            out.append(KernelArg(ordinal=ordinal, offset=offset, size=size))
        return out

    def kernel_arg_by_ordinal(self, kernel, ordinal):
        entries = self.arg_map.get(kernel)
        if not entries or ordinal < 0 or ordinal >= len(entries):
            return None
        entry = entries[ordinal]
        if entry is None:
            return None
        offset, size = entry
        return KernelArg(ordinal=ordinal, offset=offset, size=size)

    def kernel_arg_by_offset(self, kernel, offset):
        entries = self.arg_map.get(kernel)
        if not entries:
            return None
        for ordinal, entry in enumerate(entries):
            if entry is None:
                continue
            entry_offset, size = entry
            if entry_offset == offset:
                return KernelArg(ordinal=ordinal, offset=entry_offset, size=size)
        return None

    def const_words(self, kernel, bank):
        per_kernel = (self.const_data.get(kernel) or {}).get(bank)
        if per_kernel:
            return per_kernel
        # Fall back to a module-shared bank (e.g. the IDCT zigzag in bank 3,
        # emitted as a bare `.nv.constant3` with no kernel suffix).
        return self.shared_const_data.get(bank) or []

    def const_size_bytes(self, kernel, bank):
        return len(self.const_words(kernel, bank)) * 4

    def read_const_u32(self, kernel, bank, byte_offset):
        return self.read_const_int(kernel, bank, byte_offset, 4, signed=False)

    def read_const_int(
        self,
        kernel,
        bank,
        byte_offset,
        size,
        *,
        signed=False,
    ):
        if size <= 0:
            return 0
        if byte_offset < 0:
            return None

        words = (self.const_data.get(kernel) or {}).get(bank)
        if not words:
            return None

        result = 0
        for i in range(size):
            curr_byte_offset = byte_offset + i
            w_idx = curr_byte_offset // 4
            b_idx = curr_byte_offset % 4
            if w_idx < len(words):
                byte_val = (words[w_idx] >> (b_idx * 8)) & 0xFF
            else:
                byte_val = 0
            result |= byte_val << (i * 8)

        if signed:
            sign_bit = 1 << (size * 8 - 1)
            if result & sign_bit:
                result -= 1 << (size * 8)
        return result

    # ------------------------------------------------------------------
    # Binary cubin parsing (pyelftools)
    # ------------------------------------------------------------------

    def _parse_binary_cubin(self):
        """Parse a raw binary CUDA cubin using pyelftools."""
        from elftools.elf.elffile import ELFFile

        with open(self.path, 'rb') as f:
            elf = ELFFile(f)
            flags = elf.header["e_flags"]
            # CUDA ELF ABI 6 records the cubin's physical SM in bits 8..15.
            self.blackwell_param_abi = flags >> 24 == 6 and (flags >> 8) & 0xff in (100, 120)
            self._parse_binary_const_sections(elf)
            self._parse_binary_nvinfo_sections(elf)
            self._parse_binary_frame_size(elf)
            self._parse_binary_rel_sections(elf)
            self._parse_binary_const4_relocs(elf)
            self._parse_binary_nv_global(elf)
            self._parse_binary_abs32_relocs(elf)
            self._parse_binary_shared_sizes(elf)
            self._parse_guarded_tmem_entries(elf)

    def _parse_guarded_tmem_entries(self, elf):
        # CUDA 12.9 puts two runtime-owned allocation fields in reserved
        # shared memory. Only a CUDA entry starts a new lifetime for them.
        if (not self.blackwell_param_abi or elf.elfclass != 64 or
                not elf.get_section_by_name(".nv.shared.reserved.0")):
            return
        tab = elf.get_section_by_name(".symtab")
        if not tab:
            return
        required = {("__nv_reservedSMEM_allocation_phase", 0x40, 1),
                    ("__nv_reservedSMEM_allocation_mask", 0x50, 4)}
        raw = tab.data()
        entries = set()
        for i, sym in enumerate(tab.iter_symbols()):
            index = sym["st_shndx"]
            if not isinstance(index, int):
                continue
            field = (sym.name, sym["st_value"], sym["st_size"])
            if field in required and sym["st_info"]["type"] == "STT_OBJECT":
                if elf.get_section(index).name == ".nv.shared.reserved.0":
                    required.remove(field)
            # pyelftools exposes visibility but discards CUDA's STO_CUDA_ENTRY
            # flag. st_other is byte 5 of Elf64_Sym.
            other = raw[i * tab["sh_entsize"] + 5]
            if (other & 0x10 and sym["st_info"]["type"] == "STT_FUNC" and sym["st_value"] == 0 and
                    elf.get_section(index).name == ".text." + sym.name):
                entries.add(sym.name)
        if not required:
            self.guarded_tmem_entries = entries

    def _parse_binary_shared_sizes(self, elf):
        """Per-kernel static shared size from `.nv.shared.<kernel>` (NOBITS)
        section headers — the binary-cubin twin of _parse_shared_sizes.
        Without this, raw-cubin inputs (FA3 kernel.cubin, sm90_synth
        microbench cubins) reported max_shared_size()==0 even for kernels
        with real static __shared__, breaking the dynamic-shared
        discriminator both ways. NOTE: on sm_90 the recorded size includes
        the 1 KiB reserved SW region (e.g. 0x1c00 for 6 KiB of user arrays;
        0x400 for a zero-static kernel), matching the window-addressed SASS
        offsets, so sizing @shared_mem to it is exactly right."""
        for sec in elf.iter_sections():
            name = sec.name or ""
            if not name.startswith(".nv.shared."):
                continue
            kernel = name[len(".nv.shared."):]
            size = sec['sh_size']
            self.shared_sizes[kernel] = max(
                self.shared_sizes.get(kernel, 0), size)

    def _parse_binary_frame_size(self, elf):
        """Extract EIATTR_FRAME_SIZE per kernel from the module-level .nv.info.

        The section is a stream of (format, attribute, length, value) records.
        EIATTR_FRAME_SIZE (0x11) is an SVAL (format 0x04) carrying 8 bytes:
          - u32 symbol index into .symtab (identifies the kernel)
          - u32 frame size in bytes

        The lifter materialises this value into c[0x0][0x28] (see
        `Lifter._load_const_from_elf_const_data`), so the SASS prologue
        `LDC R1, c[0x0][0x28]; VIADD R, R1, -frame_size` resolves to a sane
        local-memory base instead of leaving R1 at 0 and underflowing.
        """
        import struct

        EIFMT_SVAL = 0x04
        EIFMT_HVAL = 0x03
        EIFMT_BVAL = 0x02
        EIFMT_NVAL = 0x01
        EIATTR_FRAME_SIZE = 0x11
        EIATTR_MIN_STACK_SIZE = 0x12

        symtab = elf.get_section_by_name('.symtab')
        nv_info = elf.get_section_by_name('.nv.info')
        if not symtab or not nv_info:
            return
        data = nv_info.data()
        i = 0
        while i + 4 <= len(data):
            fmt = data[i]
            attr = data[i + 1]
            length = data[i + 2] | (data[i + 3] << 8)
            i += 4
            if fmt == EIFMT_SVAL:
                if attr == EIATTR_FRAME_SIZE and length >= 8 and i + 8 <= len(data):
                    symidx, framesize = struct.unpack('<II', data[i:i + 8])
                    sym = symtab.get_symbol(symidx)
                    if sym and framesize:
                        self.frame_size[sym.name] = framesize
                elif attr == EIATTR_MIN_STACK_SIZE and length >= 8 and i + 8 <= len(data):
                    # Total stack depth (own frame + deepest callee chain) — used
                    # as the c[0x0][0x28] SP base after device-fn inlining.
                    symidx, stacksize = struct.unpack('<II', data[i:i + 8])
                    sym = symtab.get_symbol(symidx)
                    if sym and stacksize:
                        self.min_stack_size[sym.name] = stacksize
                i += length
            elif fmt in (EIFMT_HVAL, EIFMT_BVAL, EIFMT_NVAL):
                # value (if any) is encoded in the header bytes — already consumed.
                pass
            else:
                break

    def _parse_binary_const_sections(self, elf):
        """Extract .nv.constant<bank>[.<kernel>] sections from binary ELF.

        The `.<kernel>` suffix is optional: a bare `.nv.constant<N>` is a
        module-shared bank (e.g. the IDCT zigzag in bank 3)."""
        const_re = re.compile(r'^\.nv\.constant(\d+)(?:\.(.+))?$')
        for section in elf.iter_sections():
            m = const_re.match(section.name)
            if not m:
                continue
            bank = int(m.group(1))
            kernel = m.group(2)  # None for a shared (no-suffix) bank
            data = section.data()
            if not data:
                continue
            n_words = len(data) // 4
            words = list(struct.unpack(f'<{n_words}I', data[:n_words * 4]))
            if kernel is None:
                self.shared_const_data[bank] = words
            else:
                self.const_data.setdefault(kernel, {})[bank] = words

    def _parse_binary_nvinfo_sections(self, elf):
        """Extract kernel param info from .nv.info.<kernel> sections.

        The per-kernel section is a stream of (format, attribute, data)
        records:
          - format 0x01/0x02/0x03 (NVAL/BVAL/HVAL): 4 bytes total, the
            u16 at +2 is the inline value.
          - format 0x04 (SVAL): u16 at +2 is the payload length, payload
            follows.
        EIATTR_KPARAM_INFO (0x17, SVAL, 12 bytes):
          u32 index, u16 ordinal, u16 offset, u32 word with
          size = (word >> 18) & 0x3fff.
        EIATTR_PARAM_CBANK (0x0a, SVAL, 8 bytes):
          u32 section index, u32 word with param base = word & 0xffff
          (0x210 on sm_90).
        EIATTR_MAX_THREADS (0x05, SVAL, 12 bytes): u32 x, y, z block dims
          — populates max_threads like the cuobjdump-text parser (binary
          cubins previously left it empty, so cudnn BD=512 kernels were
          launched with the 256 default and hung).
        Populates arg_map identically to the cuobjdump-text parser:
        kernel -> [(offset + param_base, size)] indexed by ordinal.
        """
        import struct

        for section in elf.iter_sections():
            if not section.name.startswith('.nv.info.'):
                continue
            kernel = section.name[len('.nv.info.'):]
            data = section.data()
            param_base = 0
            pairs = []
            i = 0
            while i + 4 <= len(data):
                fmt, attr = data[i], data[i + 1]
                val16 = struct.unpack_from('<H', data, i + 2)[0]
                if fmt == 0x04:  # EIFMT_SVAL
                    payload = data[i + 4:i + 4 + val16]
                    if attr == 0x17 and len(payload) >= 12:  # KPARAM_INFO
                        _, ordinal, offset, word = struct.unpack_from(
                            '<IHHI', payload)
                        pairs.append((ordinal, offset,
                                      (word >> 18) & 0x3fff))
                    elif attr == 0x0a and len(payload) >= 8:  # PARAM_CBANK
                        word = struct.unpack_from('<I', payload, 4)[0]
                        param_base = word & 0xffff
                        self.param_cbank_base[kernel] = param_base
                    elif attr == 0x05 and len(payload) >= 12:  # MAX_THREADS
                        self.max_threads[kernel] = struct.unpack_from(
                            '<3I', payload)
                    elif attr == 0x3d and len(payload) == 12:  # CTA_PER_CLUSTER
                        dims = struct.unpack('<3I', payload)
                        if all(dims):
                            self.cluster_dims[kernel] = dims
                    i += 4 + val16
                else:  # NVAL/BVAL/HVAL: value inline, fixed 4 bytes
                    i += 4
            # A `.nv.info.<kernel>` section exists only for kernel entry
            # points (device functions like $__internal_* have none), so its
            # presence — not the presence of KPARAM_INFO records — is what
            # marks a kernel. A zero-param kernel (e.g. dispatch-cuda's
            # `_Z11EmptyKernelv`) has no KPARAM_INFO at all; it must still
            # land in arg_map (with an empty list) or the nvvm lifter never
            # marks it as a kernel and llc emits `.func` instead of `.entry`,
            # making the lifted PTX unusable for kernel launch (H200 pilot
            # 142532 NO_SWAP).
            if not pairs:
                self.arg_map[kernel] = []
                continue
            pairs.sort()
            lst = [None] * (pairs[-1][0] + 1)
            for ordinal, offset, size in pairs:
                lst[ordinal] = (offset + param_base, size)
            self.arg_map[kernel] = lst

    def _parse_binary_rel_sections(self, elf):
        """Extract .rel.text.<kernel> relocation entries from binary ELF."""
        symtab = None
        for section in elf.iter_sections():
            if section.name == '.symtab':
                symtab = section
                break

        rel_re = re.compile(r'^\.rel\.text\.(.+)$')
        for section in elf.iter_sections():
            m = rel_re.match(section.name)
            if not m:
                continue
            kernel = m.group(1)
            # Skip RELA sections (different format)
            if section['sh_type'] not in ('SHT_REL',):
                continue
            mapping = {}
            if symtab:
                for rel in section.iter_relocations():
                    offset = rel['r_offset']
                    sym_idx = rel['r_info_sym']
                    sym = symtab.get_symbol(sym_idx)
                    if sym:
                        mapping[offset] = sym.name
            if mapping:
                self.rel_text_call_map[kernel] = mapping

    def _parse_binary_const4_relocs(self, elf):
        """Extract .rela.nv.constant4 entries: c[0x4] slot -> symbol."""
        symtab = None
        for section in elf.iter_sections():
            if section.name == '.symtab':
                symtab = section
                break
        if symtab is None:
            return
        for section in elf.iter_sections():
            if section.name != '.rela.nv.constant4':
                continue
            if not section.is_RELA():
                continue
            for rel in section.iter_relocations():
                sym = symtab.get_symbol(rel['r_info_sym'])
                if not sym or not sym.name:
                    continue
                defined = sym['st_shndx'] != 'SHN_UNDEF'
                value = sym['st_value'] if defined else None
                is_func = sym['st_info']['type'] == 'STT_FUNC'
                self.const4_relocs[rel['r_offset']] = (sym.name, value, is_func)

    def _parse_binary_nv_global(self, elf):
        """Extract .nv.global(.init) data and their OBJECT symbols."""
        init_idx = bss_idx = None
        for idx, sec in enumerate(elf.iter_sections()):
            if sec.name == '.nv.global.init':
                init_idx = idx
                self.nv_global_init = sec.data()
            elif sec.name == '.nv.global':
                bss_idx = idx
                self.nv_global_size = sec['sh_size']
        if init_idx is None and bss_idx is None:
            return
        symtab = elf.get_section_by_name('.symtab')
        if symtab is None:
            return
        for sym in symtab.iter_symbols():
            if not sym.name or sym['st_info']['type'] != 'STT_OBJECT':
                continue
            shndx = sym['st_shndx']
            if shndx == init_idx:
                self.nv_global_syms[sym.name] = (
                    True, sym['st_value'], sym['st_size'])
            elif shndx == bss_idx:
                self.nv_global_syms[sym.name] = (
                    False, sym['st_value'], sym['st_size'])

    # ELF relocation types for CUDA ABS32 immediate patching (the loader
    # writes the low/high half of a symbol's address into the 32-bit
    # immediate field of the tagged instruction).
    R_CUDA_ABS32_LO_32 = 56
    R_CUDA_ABS32_HI_32 = 57

    # CUDA CALL-target relocation against a function symbol. A
    # `CALL.ABS.NOINC 0x0` whose callee lives in another top-level `.text`
    # section is patched by the loader via this relocation; the relocated
    # offset is the CALL instruction's text offset and the symbol names the
    # callee function. (Numeric type 50 / textual `R_CUDA_ABS55_16_34`.)
    R_CUDA_ABS55_16_34 = 50

    def _is_call_target_reloc(self, rtype):
        """True for relocation types that name a CALL.ABS.NOINC callee."""
        if isinstance(rtype, int):
            return rtype == self.R_CUDA_ABS55_16_34
        return str(rtype) == 'R_CUDA_ABS55_16_34'

    def _parse_binary_abs32_relocs(self, elf):
        """Extract R_CUDA_ABS32_LO/HI_32 entries from .rela.text.<kernel>."""
        symtab = elf.get_section_by_name('.symtab')
        if symtab is None:
            return
        rela_re = re.compile(r'^\.rela\.text\.(.+)$')
        for section in elf.iter_sections():
            m = rela_re.match(section.name)
            if not m or not section.is_RELA():
                continue
            kernel = m.group(1)
            mapping = {}
            call_mapping = {}
            for rel in section.iter_relocations():
                rtype = rel['r_info_type']
                sym = symtab.get_symbol(rel['r_info_sym'])
                if not sym or not sym.name:
                    continue
                # Mask to the 16-byte instruction start: the reloc may
                # address the immediate field inside the instruction.
                off = rel['r_offset'] & ~0xF
                if rtype in (self.R_CUDA_ABS32_LO_32, self.R_CUDA_ABS32_HI_32):
                    mapping[off] = (sym.name, rtype == self.R_CUDA_ABS32_HI_32)
                elif self._is_call_target_reloc(rtype):
                    call_mapping[off] = sym.name
            if mapping:
                self.rel_text_abs32[kernel] = mapping
            if call_mapping:
                self.rel_text_call_map.setdefault(kernel, {}).update(
                    call_mapping)

    # ------------------------------------------------------------------
    # Text dump parsing (cuobjdump -elf output)
    # ------------------------------------------------------------------

    def _parse(self, text):
        self._parse_arg_map(text)
        self._parse_const_data(text)
        self._parse_rel_text_call_map(text)
        self._parse_frame_size(text)
        self._parse_min_stack_size(text)
        self._parse_launch_dims(text)
        self._parse_text_symtab(text)
        self._parse_nv_global(text)
        self._parse_rela_tables(text)
        self._parse_shared_sizes(text)

    def _parse_shared_sizes(self, text):
        """Per-kernel static shared size from `.nv.shared.<kernel>` (NOBITS).
        Section-table row: `idx off size es align TYPE flags link info name`
        (all hex, no 0x). size is column 3; name carries the mangled kernel."""
        sec_row = re.compile(
            r"^\s*([0-9a-f]+)\s+([0-9a-f]+)\s+([0-9a-f]+)\s+[0-9a-f]+\s+"
            r"[0-9a-f]+\s+(\S+)\s+[0-9a-f]+\s+[0-9a-f]+\s+[0-9a-f]+\s+"
            r"(\.nv\.shared\.\S+)\s*$")
        for line in text.splitlines():
            m = sec_row.match(line)
            if not m:
                continue
            size = int(m.group(3), 16)
            kernel = m.group(5)[len(".nv.shared."):]
            self.shared_sizes[kernel] = max(self.shared_sizes.get(kernel, 0), size)

    def max_shared_size(self):
        """Largest static shared usage across the module's kernels (bytes)."""
        return max(self.shared_sizes.values(), default=0)

    def static_shared_bytes(self, kernel):
        """Per-kernel STATIC `.nv.shared.<kernel>` size in bytes (0 if none).

        DISTINCT from max_shared_size(): (a) per-kernel, not the module max, and
        (b) parsed for BINARY cubins too. `_parse_binary_cubin` populates
        arg/const/nv-info sections but NOT shared_sizes, so for a raw cubin
        max_shared_size() is 0 for every kernel — which cannot distinguish a
        kernel with a real static allocation from one that only carries the fixed
        0x400 reserved shared window (the FA3-style fully-dynamic case). This
        method reads the `.nv.shared.<kernel>` NOBITS section sizes directly so
        the NVVM lifter's FA-dynamic detector can tell them apart. It does NOT
        touch shared_sizes / max_shared_size(), so the existing sizing path is
        byte-for-byte unchanged.

        Lazily parsed and cached; safe (best-effort) on a text-dump path (no
        binary ELF -> falls back to whatever shared_sizes the text parser found).
        """
        cache = getattr(self, "_static_shared_cache", None)
        if cache is None:
            cache = {}
            try:
                from elftools.elf.elffile import ELFFile
                with open(self.path, 'rb') as f:
                    if f.read(4) == _ELF_MAGIC:
                        f.seek(0)
                        for sec in ELFFile(f).iter_sections():
                            n = sec.name
                            if (n.startswith(".nv.shared.")
                                    and not n.startswith(".nv.shared.reserved")):
                                k = n[len(".nv.shared."):]
                                cache[k] = max(cache.get(k, 0),
                                               sec.header['sh_size'])
            except Exception:
                pass
            # Text-dump path already parsed sizes into shared_sizes.
            for k, v in self.shared_sizes.items():
                cache[k] = max(cache.get(k, 0), v)
            self._static_shared_cache = cache
        return cache.get(kernel, 0)

    def max_threads_for(self, kernel):
        return self.max_threads.get(kernel)

    def _parse_launch_dims(self, text):
        # Per-kernel .nv.info.<kernel> section carries:
        #     Attribute:   EIATTR_MAX_THREADS
        #     Format:      EIFMT_SVAL
        #     Value:       0x400 0x1 0x1        (= block dims x y z)
        sec_re = re.compile(
            r"^\.nv\.info\.(\S+)\s*\n(.*?)(?=\n\.nv\.|\n\.text|\n\.rel|\Z)",
            re.DOTALL | re.MULTILINE,
        )
        val_re = re.compile(
            r"EIATTR_(MAX_THREADS|CTA_PER_CLUSTER)\s*\n\s*Format:\s*EIFMT_SVAL\s*\n\s*Value:\s*"
            r"(0x[0-9a-fA-F]+)\s+(0x[0-9a-fA-F]+)\s+(0x[0-9a-fA-F]+)"
        )
        for m in sec_re.finditer(text):
            kernel, body = m.group(1), m.group(2)
            for attribute, x, y, z in val_re.findall(body):
                dims = tuple(int(value, 16) for value in (x, y, z))
                if attribute == "MAX_THREADS":
                    self.max_threads[kernel] = dims
                elif all(dims):
                    self.cluster_dims[kernel] = dims

    def _parse_frame_size(self, text):
        # Each kernel entry in .nv.info.<kernel> has an EIATTR_FRAME_SIZE
        # attribute formatted as:
        #     Attribute:   EIATTR_FRAME_SIZE
        #     Format:      EIFMT_SVAL
        #     Value:       function: <kernel>(...)   frame size: 0xNNNN
        # We only need the integer hex after "frame size:".
        pat = re.compile(
            r"EIATTR_FRAME_SIZE.*?function:\s*([^\s\(]+)\([^\)]*\)\s*"
            r"frame size:\s*0x([0-9a-fA-F]+)",
            re.DOTALL,
        )
        for m in pat.finditer(text):
            kernel = m.group(1)
            self.frame_size[kernel] = int(m.group(2), 16)

    def _parse_min_stack_size(self, text):
        # EIATTR_MIN_STACK_SIZE records the kernel's TOTAL stack depth (own
        # frame + deepest callee chain). Format mirrors EIATTR_FRAME_SIZE:
        #     Attribute:   EIATTR_MIN_STACK_SIZE
        #     Value:       function: <kernel>(...)   min stack size: 0xNNNN
        pat = re.compile(
            r"EIATTR_MIN_STACK_SIZE.*?function:\s*([^\s\(]+)\([^\)]*\)\s*"
            r"min stack size:\s*0x([0-9a-fA-F]+)",
            re.DOTALL,
        )
        for m in pat.finditer(text):
            kernel = m.group(1)
            self.min_stack_size[kernel] = int(m.group(2), 16)

    def _parse_arg_map(self, text):
        info_pattern = re.compile(
            r"\.nv\.info\.([^\s]+)\s*(.*?)(?=\n\.|\Z)",
            re.DOTALL,
        )

        for m in info_pattern.finditer(text):
            kernel = m.group(1)
            body = m.group(2)

            # Attribute: EIATTR_PARAM_CBANK
            # Value:    0x2 0x1c0160
            param_base = 0
            cbank_match = re.search(
                r"Attribute:\s*EIATTR_PARAM_CBANK.*?Value:\s*(?:0x)?[0-9a-fA-F]+\s+(?:0x)?([0-9a-fA-F]+)",
                body,
                re.DOTALL,
            )
            if cbank_match:
                full_val = int(cbank_match.group(1), 16)
                param_base = full_val & 0xFFFF
                self.param_cbank_base[kernel] = param_base

            pairs = []
            for om in re.finditer(
                r"Ordinal\s*:\s*(?:0x)?([0-9a-fA-F]+).*?Offset\s*:\s*(?:0x)?([0-9a-fA-F]+).*?Size\s*:\s*(?:0x)?([0-9a-fA-F]+)",
                body,
                re.DOTALL,
            ):
                ordinal = int(om.group(1), 16)
                offset = int(om.group(2), 16)
                size = int(om.group(3), 16)
                pairs.append((ordinal, offset, size))

            if not pairs:
                # Zero-param kernel: still a kernel (the .nv.info.<kernel>
                # block only exists for entry points) — record it so the
                # nvvm lifter marks it `.entry`. Mirrors the binary parser.
                self.arg_map[kernel] = []
                continue

            pairs.sort()
            max_ord = pairs[-1][0]
            lst = [None] * (max_ord + 1)
            for ord_, off, sz in pairs:
                lst[ord_] = (off + param_base, sz)
            self.arg_map[kernel] = lst

    def _parse_const_data(self, text):
        # The `.<kernel>` suffix is OPTIONAL: a bare `.nv.constant<N>` is a
        # module-shared bank (e.g. bank 3 = the nvjpeg IDCT zigzag table) that
        # every kernel reading c[N][...] consumes. Without this, those tables
        # were dropped and reads returned zeros (broken IDCT / dequant).
        const_pattern = re.compile(
            r"^\.nv\.constant([0-9]+)(?:\.([^\s]+))?\s*(.*?)(?=\n\.|\Z)",
            re.DOTALL | re.MULTILINE,
        )

        for m in const_pattern.finditer(text):
            bank = int(m.group(1))
            kernel = m.group(2)  # None for a shared (no-suffix) bank
            body = m.group(3)

            words = [int(h, 16) for h in re.findall(r"(?:0x)?([0-9a-fA-F]+)", body)]

            if kernel is None:
                self.shared_const_data[bank] = words
            else:
                kernel_banks = self.const_data.setdefault(kernel, {})
                kernel_banks[bank] = words

    def rel_text_call_targets(self, function):
        return self.rel_text_call_map.get(function) or {}

    def rel_text_call_target(self, function, offset):
        if isinstance(offset, str):
            offset = offset.strip().lower()
            if offset.startswith("0x"):
                offset = int(offset, 16)
            else:
                offset = int(offset, 0)
        return (self.rel_text_call_map.get(function) or {}).get(offset)

    # cuobjdump dumps a fatbin as several concatenated ELF blocks, each
    # preceded by this marker. Section indices and the .symtab restart per
    # block, so device-global symbols must be resolved against the section
    # table of the SAME block.
    _ELF_BLOCK_RE = re.compile(r"^Fatbin elf code:\s*$", re.MULTILINE)

    def _parse_nv_global(self, text):
        """Extract .nv.global(.init) sections + symbols from a text dump.

        The cuobjdump -elf header table gives section index/size/type:
            Index Offset   Size ES Align  Type  Flags Link Info Name
                a    800     20  0  4  PROGBITS    3    0    0 .nv.global.init
                b    820     10  0  4    NOBITS    3    0    0 .nv.global
        (all numbers hex, no 0x). The .nv.global.init body appears later as
        hex words; .nv.global (NOBITS) has no body — size comes from here.

        Resolve section indices and symbols PER ELF block: a fatbin holds
        several concatenated ELF dumps (e.g. perlin's CUDA math div helper
        precedes the user module), each with its own section numbering and
        .symtab. A global scan would match block 1's symtab against block
        3's .nv.global.init index, leaving nv_global_syms empty and every
        device-global table base reading 0 (-> null deref / garbage).
        """
        sec_row = re.compile(
            r"^\s*([0-9a-f]+)\s+([0-9a-f]+)\s+([0-9a-f]+)\s+[0-9a-f]+\s+"
            r"[0-9a-f]+\s+(\S+)\s+[0-9a-f]+\s+[0-9a-f]+\s+[0-9a-f]+\s+(\S+)\s*$")
        for block in self._ELF_BLOCK_RE.split(text):
            init_idx = bss_idx = None
            bss_size = 0
            for line in block.splitlines():
                m = sec_row.match(line)
                if not m:
                    continue
                idx, size, name = int(m.group(1), 16), int(m.group(3), 16), m.group(5)
                if name == '.nv.global.init':
                    init_idx = idx
                elif name == '.nv.global':
                    bss_idx = idx
                    bss_size = size
            if init_idx is None and bss_idx is None:
                continue

            # Base offsets so multiple contributing blocks concatenate
            # cleanly; identity for the common single-block case.
            init_base = len(self.nv_global_init)
            bss_base = self.nv_global_size
            body_m = re.search(
                r"^\.nv\.global\.init\s*\n(.*?)(?=\n\.|\Z)",
                block, re.DOTALL | re.MULTILINE)
            if body_m:
                words = [int(h, 16) for h in
                         re.findall(r"0x([0-9a-fA-F]+)", body_m.group(1))]
                self.nv_global_init += b"".join(
                    struct.pack('<I', w & 0xFFFFFFFF) for w in words)
            if bss_size:
                self.nv_global_size += bss_size

            for name, (value, size, info, shndx) in \
                    self._parse_symtab_block(block).items():
                if info & 0xF != 1:  # STT_OBJECT only
                    continue
                if init_idx is not None and shndx == init_idx:
                    self.nv_global_syms[name] = (True, init_base + value, size)
                elif bss_idx is not None and shndx == bss_idx:
                    self.nv_global_syms[name] = (False, bss_base + value, size)

    def _parse_symtab_block(self, block):
        """Parse a single ELF block's .symtab into
        {name: (value, size, info, shndx)}. Columns are
        `index value size info other shndx name`, mixed decimal-0/0xNN."""
        syms = {}
        sym_m = re.search(
            r"^\.section\s+\.symtab\s*\n(.*?)(?=\n\.\S|\Z)",
            block, re.DOTALL | re.MULTILINE)
        if not sym_m:
            return syms
        sym_row = re.compile(
            r"^\s*(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s*$")
        for line in sym_m.group(1).splitlines():
            m = sym_row.match(line)
            if not m or m.group(1) == 'index' or m.group(7) == '(null)':
                continue
            try:
                syms[m.group(7)] = (
                    int(m.group(2), 0), int(m.group(3), 0),
                    int(m.group(4), 0), int(m.group(6), 0))
            except ValueError:
                continue
        return syms

    def _parse_text_symtab(self, text):
        """Merge every ELF block's .symtab into _text_syms. A fatbin dump
        has one .symtab per block; reading only the first (re.search) missed
        device-global and relocation symbols defined in later blocks."""
        self._text_syms = {}
        for block in self._ELF_BLOCK_RE.split(text):
            self._text_syms.update(self._parse_symtab_block(block))

    def _parse_rela_tables(self, text):
        """Parse RELA tables from a text dump:
            .section .rela.text.<kernel>  -> rel_text_abs32 (ABS32 LO/HI)
            .section .rela.nv.constant4   -> const4_relocs  (R_CUDA_64)
        Entry rows: `0x<off>  <symbol>  <R_CUDA_*>  0x<addend>`.
        """
        header_re = re.compile(r"^\.section\s+(\.rela\.[^\s]+)\s+RELA\s*$")
        entry_re = re.compile(
            r"^(?:0x)?([0-9a-fA-F]+)\s+(\S+)\s+(R_CUDA_\S+)")
        lines = text.splitlines()
        i = 0
        while i < len(lines):
            m = header_re.match(lines[i])
            if not m:
                i += 1
                continue
            sec_name = m.group(1)
            i += 1
            entries = []
            while i < len(lines) and not lines[i].startswith(".section"):
                em = entry_re.match(lines[i].strip())
                if em:
                    entries.append((int(em.group(1), 16),
                                    em.group(2), em.group(3)))
                i += 1

            if sec_name == '.rela.nv.constant4':
                for off, sym, rtype in entries:
                    if rtype != 'R_CUDA_64':
                        continue
                    info = getattr(self, '_text_syms', {}).get(sym)
                    if info is not None:
                        value, _size, sinfo, shndx = info
                        self.const4_relocs.setdefault(
                            off, (sym, value if shndx != 0 else None,
                                  (sinfo & 0xF) == 2))
                    else:
                        self.const4_relocs.setdefault(off, (sym, None, False))
            elif sec_name.startswith('.rela.text.'):
                kernel = sec_name[len('.rela.text.'):]
                mapping = {}
                call_mapping = {}
                for off, sym, rtype in entries:
                    if rtype == 'R_CUDA_ABS32_LO_32':
                        mapping[off & ~0xF] = (sym, False)
                    elif rtype == 'R_CUDA_ABS32_HI_32':
                        mapping[off & ~0xF] = (sym, True)
                    elif self._is_call_target_reloc(rtype):
                        # Cross-function CALL target: `CALL.ABS.NOINC 0x0`
                        # whose real callee is a separate top-level Function
                        # symbol the loader patches in. The relocation offset
                        # is the CALL instruction's text offset; record it so
                        # HandleCall / CrossFuncInline can resolve the callee.
                        call_mapping[off & ~0xF] = sym
                if mapping:
                    self.rel_text_abs32[kernel] = mapping
                if call_mapping:
                    self.rel_text_call_map.setdefault(kernel, {}).update(
                        call_mapping)

    def _parse_rel_text_call_map(self, text):
        lines = text.splitlines()
        i = 0
        header_re = re.compile(r"^\.section\s+\.rel\.text\.([^\s]+)\s+REL\s*$")
        entry_re = re.compile(r"^(?:0x)?([0-9a-fA-F]+)\s+([^\s]+)")

        while i < len(lines):
            m = header_re.match(lines[i])
            if not m:
                i += 1
                continue

            function = m.group(1)
            i += 1
            mapping = {}

            while i < len(lines) and not lines[i].startswith(".section"):
                line = lines[i].strip()
                if line:
                    em = entry_re.match(line)
                    if em:
                        offset = int(em.group(1), 16)
                        target = em.group(2)
                        mapping[offset] = target
                i += 1

            if mapping:
                self.rel_text_call_map[function] = mapping
