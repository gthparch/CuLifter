/* Host-runtime device printf for lifted kernels (x86 backend).
 *
 * CUDA device printf compiles to a SASS CALL to `vprintf(const char *fmt,
 * void *packed_args)` — the second argument is NOT a va_list but a buffer
 * the kernel packs each argument into at its natural alignment (ints 4B;
 * long/size_t/pointers/%s 8B; all floats promoted to double, 8B; %c packed
 * as int). FlashAttention cubins import this symbol.
 *
 * Plain C on purpose: <cstdio>'s `vprintf(const char*, va_list)` declaration
 * makes this signature ill-formed in any C++ TU (kernel_wrapper_75.h
 * documents this). Compiled separately and linked in, our definition takes
 * precedence over libc's at link time. No stdio header is included, so the
 * libc prototype never clashes; we declare the two functions we need.
 *
 * Format handling covers the printf subset CUDA documents (no %n, no
 * positional args). Unknown conversions are echoed literally.
 */

/* The mismatched-vs-libc signatures below are the whole point (see header
 * comment); quiet the compiler's builtin checks for exactly these. */
#ifdef __clang__
#pragma clang diagnostic ignored "-Wincompatible-library-redeclaration"
#pragma clang diagnostic ignored "-Wbuiltin-requires-header"
#endif

typedef __SIZE_TYPE__ size_t;
extern int fputs(const char *s, void *stream);
extern int fprintf(void *stream, const char *fmt, ...);
extern void *stdout;

static const char *pack_align(const char *p, size_t a)
{
    size_t v = (size_t)p;
    v = (v + a - 1) & ~(a - 1);
    return (const char *)v;
}

/* Lifted kernels pass the packed-args buffer as a LOCAL-memory generic
 * address: R1(=0 in the wrapper) + offset + local-window-base(c[0][0x20],
 * also 0). That reaches us as a small integer offset into the thread's
 * local_mem, not a host pointer — translate it (CUTLASS 02_dump_reg_shmem
 * printf("%.0f", v) packs v at [R1+0x8] and passes 0x8). Offset 0 is VALID
 * (dump_shmem's "TB %d" packs bid at [R1+0x0]) — do not treat it as NULL;
 * no-arg printfs that pass 0 never read the buffer anyway. Host pointers
 * (c[0x4]-materialized) are far above 32KB and pass through untouched.
 * Same boundary rule as is_local_address() in kernel_wrapper_75.h. */
extern _Thread_local unsigned char local_mem[32768];

int vprintf(const char *fmt, void *packed)
{
    if ((size_t)packed < sizeof(local_mem))
        packed = (void *)(local_mem + (size_t)packed);
    const char *arg = (const char *)packed;
    char spec[32];
    char buf[2];
    int written = 0;

    if (!fmt)
        return -1;

    while (*fmt) {
        if (*fmt != '%') {
            buf[0] = *fmt++; buf[1] = 0;
            fputs(buf, stdout);
            written++;
            continue;
        }

        /* copy "%{flags}{width}{.prec}{length}{conv}" into spec */
        const char *start = fmt++;
        while (*fmt && ((*fmt >= '0' && *fmt <= '9') ||
                        *fmt == '-' || *fmt == '+' || *fmt == ' ' ||
                        *fmt == '#' || *fmt == '.'))
            fmt++;
        int longs = 0;
        while (*fmt == 'l' || *fmt == 'h' || *fmt == 'z') {
            if (*fmt == 'l' || *fmt == 'z') longs++;
            fmt++;
        }
        char conv = *fmt ? *fmt++ : 0;
        size_t speclen = (size_t)(fmt - start);
        if (speclen >= sizeof(spec)) speclen = sizeof(spec) - 1;
        __builtin_memcpy(spec, start, speclen);
        spec[speclen] = 0;

        switch (conv) {
        case '%':
            fputs("%", stdout);
            written++;
            break;
        case 'd': case 'i': case 'u': case 'o': case 'x': case 'X': case 'c':
            if (longs) {
                long long v;
                arg = pack_align(arg, 8);
                __builtin_memcpy(&v, arg, 8); arg += 8;
                written += fprintf(stdout, spec, v);
            } else {
                int v;
                arg = pack_align(arg, 4);
                __builtin_memcpy(&v, arg, 4); arg += 4;
                written += fprintf(stdout, spec, v);
            }
            break;
        case 'f': case 'F': case 'e': case 'E': case 'g': case 'G': case 'a': {
            double v;  /* device printf promotes float to double */
            arg = pack_align(arg, 8);
            __builtin_memcpy(&v, arg, 8); arg += 8;
            written += fprintf(stdout, spec, v);
            break;
        }
        case 's': case 'p': {
            const char *v;
            arg = pack_align(arg, 8);
            __builtin_memcpy(&v, arg, 8); arg += 8;
            written += fprintf(stdout, spec, v);
            break;
        }
        default:
            /* unknown conversion: echo the spec literally */
            fputs(spec, stdout);
            written += (int)speclen;
            break;
        }
    }
    return written;
}
