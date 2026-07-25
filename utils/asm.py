import re
from pathlib import Path

from utils.elf import KernelArg


class Asm:
    def __init__(self, text, path=None):
        self.path = Path(path) if path else None
        self._text = text
        self.arg_map = {}
        self.const_data = {}
        self.rel_text_call_map = {}
        self.sections = []
        self.sass_by_func = {}
        self.lines = text.splitlines()
        self.parse(text)
        self.parse_sections(self.lines)

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
        return (self.const_data.get(kernel) or {}).get(bank) or []

    def const_size_bytes(self, kernel, bank):
        return len(self.const_words(kernel, bank)) * 4

    def read_const_u32(self, kernel, bank, byte_offset):
        return self.read_const_int(kernel, bank, byte_offset, 4, signed=False)

    def read_const_int(self, kernel, bank, byte_offset, size, *, signed=False):
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

    def parse(self, text):
        self.parse_arg_map(text)
        self.parse_const_data(text)
        self.parse_rel_text_call_map(text)

    def parse_arg_map(self, text):
        section_re = re.compile(r"^\s*\.section\s+\.nv\.info\.([^\s,]+)")
        section_header_re = re.compile(r"^\s*\.section\s")
        param_base = 0
        param_size = None
        entries = []
        state = None
        shorts = []
        current_kernel = None

        for line in text.splitlines():
            match = section_re.match(line)
            if match:
                if current_kernel and entries:
                    entries.sort(key=lambda entry: entry[1])
                    size_by_ordinal = {}
                    for idx, (ordinal, offset) in enumerate(entries):
                        if idx + 1 < len(entries):
                            size = entries[idx + 1][1] - offset
                        else:
                            size = param_size - offset
                        size_by_ordinal[ordinal] = (offset + param_base, size)

                    max_ord = max(size_by_ordinal)
                    lst = [None] * (max_ord + 1)
                    for ord_, (off, size) in size_by_ordinal.items():
                        lst[ord_] = (off, size)
                    self.arg_map[current_kernel] = lst

                current_kernel = match.group(1)
                param_base = 0
                param_size = None
                entries = []
                state = None
                shorts = []
                continue

            if current_kernel and section_header_re.match(line):
                if entries:
                    entries.sort(key=lambda entry: entry[1])
                    size_by_ordinal = {}
                    for idx, (ordinal, offset) in enumerate(entries):
                        if idx + 1 < len(entries):
                            size = entries[idx + 1][1] - offset
                        else:
                            size = param_size - offset
                        size_by_ordinal[ordinal] = (offset + param_base, size)

                    max_ord = max(size_by_ordinal)
                    lst = [None] * (max_ord + 1)
                    for ord_, (off, size) in size_by_ordinal.items():
                        lst[ord_] = (off, size)
                    self.arg_map[current_kernel] = lst

                current_kernel = None
                param_base = 0
                param_size = None
                entries = []
                state = None
                shorts = []
                continue

            if current_kernel is None:
                continue

            if "EIATTR_PARAM_CBANK" in line:
                state = "param_cbank"
                shorts = []
                continue
            if "EIATTR_CBANK_PARAM_SIZE" in line:
                state = "param_size"
                continue
            if "EIATTR_KPARAM_INFO" in line:
                state = "kparam_info"
                shorts = []
                continue

            match = re.search(r"\.short\s+0x([0-9a-fA-F]+)", line)
            if not match:
                continue
            val = int(match.group(1), 16)

            if state == "param_cbank":
                shorts.append(val)
                if len(shorts) == 2:
                    param_base, param_size = shorts
                    state = None
            elif state == "param_size":
                param_size = val
                state = None
            elif state == "kparam_info":
                shorts.append(val)
                if len(shorts) == 2:
                    ordinal, offset = shorts
                    entries.append((ordinal, offset))
                    state = None

        if current_kernel and entries:
            entries.sort(key=lambda entry: entry[1])
            size_by_ordinal = {}
            for idx, (ordinal, offset) in enumerate(entries):
                if idx + 1 < len(entries):
                    size = entries[idx + 1][1] - offset
                else:
                    size = param_size - offset
                size_by_ordinal[ordinal] = (offset + param_base, size)

            max_ord = max(size_by_ordinal)
            lst = [None] * (max_ord + 1)
            for ord_, (off, size) in size_by_ordinal.items():
                lst[ord_] = (off, size)
            self.arg_map[current_kernel] = lst

    def parse_const_data(self, text):
        section_re = re.compile(
            r"^\s*\.section\s+\.nv\.constant(?P<bank>\d+)\.(?P<kernel>[^\s,]+)"
        )
        word_re = re.compile(r"\.word\s+(.+)$")
        short_re = re.compile(r"\.short\s+(.+)$")
        byte_re = re.compile(r"\.byte\s+(.+)$")
        zero_re = re.compile(r"\.zero\s+(.+)$")

        bank = None
        kernel = None
        bytes_out = []

        for line in text.splitlines():
            m = section_re.match(line)
            if m:
                if bank is not None and kernel is not None:
                    if bytes_out:
                        while len(bytes_out) % 4 != 0:
                            bytes_out.append(0)
                        words = []
                        for i in range(0, len(bytes_out), 4):
                            word = (
                                bytes_out[i]
                                | (bytes_out[i + 1] << 8)
                                | (bytes_out[i + 2] << 16)
                                | (bytes_out[i + 3] << 24)
                            )
                            words.append(word)
                        kernel_banks = self.const_data.setdefault(kernel, {})
                        kernel_banks[bank] = words
                bank = int(m.group("bank"))
                kernel = m.group("kernel")
                bytes_out = []
                continue

            if bank is None:
                continue

            if word_re.search(line):
                values = word_re.search(line).group(1).split(",")
                for value in values:
                    v = int(value.strip(), 0)
                    bytes_out.extend([(v >> (8 * i)) & 0xFF for i in range(4)])
                continue

            if short_re.search(line):
                values = short_re.search(line).group(1).split(",")
                for value in values:
                    v = int(value.strip(), 0)
                    bytes_out.extend([v & 0xFF, (v >> 8) & 0xFF])
                continue

            if byte_re.search(line):
                values = byte_re.search(line).group(1).split(",")
                for value in values:
                    v = int(value.strip(), 0)
                    bytes_out.append(v & 0xFF)
                continue

            if zero_re.search(line):
                count = int(zero_re.search(line).group(1).strip(), 0)
                bytes_out.extend([0] * count)
                continue

        if bank is not None and kernel is not None:
            if bytes_out:
                while len(bytes_out) % 4 != 0:
                    bytes_out.append(0)
                words = []
                for i in range(0, len(bytes_out), 4):
                    word = (
                        bytes_out[i]
                        | (bytes_out[i + 1] << 8)
                        | (bytes_out[i + 2] << 16)
                        | (bytes_out[i + 3] << 24)
                    )
                    words.append(word)
                kernel_banks = self.const_data.setdefault(kernel, {})
                kernel_banks[bank] = words

    def parse_rel_text_call_map(self, text):
        lines = text.splitlines()
        i = 0
        header_re = re.compile(r"^\s*\.section\s+\.rel\.text\.([^\s]+)\s+REL\s*$")
        entry_re = re.compile(r"^\s*(?:0x)?([0-9a-fA-F]+)\s+([^\s]+)")

        while i < len(lines):
            m = header_re.match(lines[i])
            if not m:
                i += 1
                continue

            function = m.group(1)
            i += 1
            mapping = {}
            while i < len(lines):
                entry = entry_re.match(lines[i])
                if not entry:
                    break
                offset = int(entry.group(1), 16)
                target = entry.group(2)
                mapping[offset] = target
                i += 1
            self.rel_text_call_map[function] = mapping

    def parse_sections(self, lines):
        section_re = re.compile(r"^\s*\.section\s+\.text\.([A-Za-z0-9_\.]+)")
        instr_re = re.compile(r"^\s*/\*[0-9a-fA-F]+\*/")
        idx = 0
        while idx < len(lines):
            match = section_re.match(lines[idx])
            if not match:
                idx += 1
                continue
            name = match.group(1)
            start_idx = idx
            idx += 1
            while idx < len(lines) and not section_re.match(lines[idx]):
                idx += 1
            end_idx = idx
            segment = lines[start_idx:end_idx]
            first_instr = None
            last_instr = None
            for seg_idx, line in enumerate(segment):
                if instr_re.match(line):
                    if first_instr is None:
                        first_instr = seg_idx
                    last_instr = seg_idx
            if first_instr is None:
                header = segment
                footer = []
            else:
                header = segment[:first_instr]
                footer = segment[last_instr + 1 :]
            self.sections.append(
                {
                    "name": name,
                    "start": start_idx,
                    "end": end_idx,
                    "header": header,
                    "footer": footer,
                }
            )
