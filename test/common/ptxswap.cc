// libptxswap.so — LD_PRELOAD shim that swaps lifted PTX in for the
// original cubin embedded in a HeCBench-style binary, without
// modifying the binary's source.
//
// Usage:
//     env LD_PRELOAD=./libptxswap.so PTXSWAP_PTX=./lifted.ptx ./hecbench_app
//
// How it works:
//
//  1. nvcc's host-side fatbin registration runs in startup constructors
//     and calls __cudaRegisterFunction(handle, host_stub, name, ...) once
//     per kernel. We intercept that call and record the (stub -> name)
//     mapping, so we know what kernel each host_stub launches.
//
//  2. We intercept cudaLaunchKernel. For each (stub, kernel_name) pair
//     we look up the matching CUfunction in our lifted PTX module. If
//     present, we dispatch via cuLaunchKernel; if not, we fall through
//     to the original cudaLaunchKernel — so kernels the lifter doesn't
//     yet handle still run from the original cubin.
//
// Set PTXSWAP_VERBOSE=1 to print one line per launch.

#define _GNU_SOURCE

#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <string>
#include <unordered_map>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>

namespace {

CUmodule g_lifted_module = nullptr;
bool g_verbose = false;

// stub_pointer -> mangled kernel name, populated by our
// __cudaRegisterFunction interceptor.
std::unordered_map<const void*, std::string> g_stub_to_name;

// stub_pointer -> resolved CUfunction in lifted module, or nullptr to
// indicate "fall through". Populated lazily on first launch.
std::unordered_map<const void*, CUfunction> g_stub_cache;

// PTXSWAP_RENAME: comma-separated substrings ("needles"). When a host
// kernel's exact mangled name is absent from the lifted module and the
// name contains a needle, the launch is remapped to the UNIQUE lifted
// kernel whose name also contains that needle. Escape hatch for version
// drift between a captured cubin and the host binary's own template
// instantiations (e.g. FA3 hdim96: the host's BwdPostprocessConvertdQ
// instantiation differs from the captured module's, but only the
// captured one decodes the captured bwd kernel's dQaccum layout).
std::vector<std::string> g_rename_needles;

// __constant__ symbol support (Design A1, d3q19-bgk):
//
// Host code fills __constant__ arrays at runtime via
// cudaMemcpyToSymbol(C_dirs, ...). Those writes land in the resident
// VANILLA cubin's constant banks; the LIFTED module bakes the bank as
// the (zero) compile-time image into one flat `@const_mem` global. We
// mirror each such host write into @const_mem at the symbol's byte
// offset so lifted kernels read the runtime data instead of zeros.
//
// host_var_ptr -> device symbol name, recorded by __cudaRegisterVar.
std::unordered_map<const void*, std::string> g_var_to_name;

// device symbol name -> (bank, byte offset within the bank), loaded
// from the PTXSWAP_SYMMAP sidecar file (one `name bank offset` per
// line). Empty when PTXSWAP_SYMMAP is unset — then the
// cudaMemcpyToSymbol interceptor is a pure pass-through and CANNOT
// change behavior for any other test.
struct SymLoc { unsigned bank; size_t offset; };
std::unordered_map<std::string, SymLoc> g_symmap;

// const_mem byte = bank * g_bank_size + symbol_offset + copy_offset.
// Derived from the lifted module's const_mem size / NUM_CONST_MEM_BANKS
// (robust against per-kernel bank-size differences). 0 until resolved.
size_t g_bank_size = 0;
const unsigned NUM_CONST_MEM_BANKS = 5;

// __device__ global symbol support (hungarian-cuda, collision-cuda):
//
// These benchmarks share state with the host through __device__ (and
// __managed__) globals: slack, cover_row, n_matches, ... The host reads
// some via cudaMemcpyFromSymbol AND some via DIRECT managed-memory access
// (n_matches, zeros_size, goto_5, repeat_kernel drive the host control
// loop). The lifter materialises ALL such globals into ONE flat
// addrspace(1) buffer (`__slifter_nv_global` for the NOBITS/.nv.global
// BSS, `__slifter_nv_global_init` for the PROGBITS/.nv.global.init
// section) at fixed byte offsets — DISJOINT from the vanilla cubin's
// symbols the host sees.
//
// We can't intercept the host's direct managed access, so instead we
// SYNC the two buffers around every lifted launch: before the kernel,
// copy each device global vanilla-symbol -> lifted buffer (so the kernel
// sees host writes / prior managed updates); after, copy lifted buffer
// -> vanilla symbol (so the host's managed reads and cudaMemcpyFromSymbol
// see the kernel's results). The vanilla device-symbol address is
// obtained via cudaGetSymbolAddress on the registered hostVar.
//
// device symbol name -> (in_init, byte offset in flat buffer, size),
// loaded from the PTXSWAP_DEVMAP sidecar (`name init_flag offset size`).
// Empty when PTXSWAP_DEVMAP is unset -> pure pass-through.
struct DevLoc { bool in_init; size_t offset; size_t size; };
std::unordered_map<std::string, DevLoc> g_devmap;
// name -> vanilla device-symbol address (resolved lazily from hostVar).
std::unordered_map<std::string, CUdeviceptr> g_vanilla_sym_addr;
// __device__ globals: deviceName -> hostVar (for cudaGetSymbolAddress).
std::unordered_map<std::string, const void*> g_name_to_hostvar;
// __managed__ globals are registered via __cudaRegisterManagedVar with a
// `void** hostVarPtrAddress`; *hostVarPtrAddress is the managed unified
// pointer the host reads/writes directly. Record the void** so we can
// dereference it at sync time (the runtime fills it during registration).
std::unordered_map<std::string, void**> g_name_to_managed_ptr;

using cuda_launch_t = cudaError_t (*)(const void*, dim3, dim3, void**,
                                      size_t, cudaStream_t);
cuda_launch_t real_cudaLaunchKernel = nullptr;

// Internal nvcc/CUDA-runtime ABI. Stable across CUDA versions.
using register_fn_t = void (*)(void**, const char*, char*, const char*,
                               int, void*, void*, void*, void*, int*);
register_fn_t real_cudaRegisterFunction = nullptr;

// __cudaRegisterVar(void **fatCubinHandle, char *hostVar,
//   char *deviceAddress, const char *deviceName, int ext, size_t size,
//   int constant, int global). Stable runtime ABI (same provenance as
// __cudaRegisterFunction, already trusted in this shim).
using register_var_t = void (*)(void**, char*, char*, const char*,
                                int, size_t, int, int);
register_var_t real_cudaRegisterVar = nullptr;

using memcpy_to_symbol_t = cudaError_t (*)(const void*, const void*,
                                           size_t, size_t,
                                           cudaMemcpyKind);
memcpy_to_symbol_t real_cudaMemcpyToSymbol = nullptr;

void log(const char* fmt, ...) {
    if (!g_verbose) return;
    va_list ap;
    va_start(ap, fmt);
    fputs("[ptxswap] ", stderr);
    vfprintf(stderr, fmt, ap);
    fputc('\n', stderr);
    va_end(ap);
}

// Load the device-symbol -> (bank, offset) map from the sidecar named
// by PTXSWAP_SYMMAP. Each line is `name bank offset` (offset within the
// bank, decimal or 0x-hex). If the env var is unset or the file is
// absent/empty, g_symmap stays empty — making the cudaMemcpyToSymbol
// interceptor a pure pass-through (no regression for tests without a
// symmap).
void load_symmap() {
    const char* path = getenv("PTXSWAP_SYMMAP");
    if (!path) return;
    FILE* f = fopen(path, "r");
    if (!f) {
        log("PTXSWAP_SYMMAP=%s could not be opened — ignoring", path);
        return;
    }
    char name[1024];
    unsigned long bank;
    unsigned long long off;
    // strtoull on the offset token handles both decimal and 0x-hex.
    char line[2048];
    while (fgets(line, sizeof(line), f)) {
        char offtok[256];
        if (sscanf(line, "%1023s %lu %255s", name, &bank, offtok) == 3) {
            off = strtoull(offtok, nullptr, 0);
            g_symmap[name] = SymLoc{static_cast<unsigned>(bank),
                                    static_cast<size_t>(off)};
        }
    }
    fclose(f);
    log("loaded %zu constant-symbol map entries from %s",
        g_symmap.size(), path);
}

// Load the __device__-global symbol map (`name init_flag offset size`
// per line) named by PTXSWAP_DEVMAP. init_flag is 1 for .nv.global.init
// (PROGBITS) symbols, 0 for .nv.global (NOBITS/BSS). Pure pass-through
// when unset/absent.
void load_devmap() {
    const char* path = getenv("PTXSWAP_DEVMAP");
    if (!path) return;
    FILE* f = fopen(path, "r");
    if (!f) {
        log("PTXSWAP_DEVMAP=%s could not be opened — ignoring", path);
        return;
    }
    char line[2048];
    char name[1024];
    unsigned long in_init;
    unsigned long long off, sz;
    while (fgets(line, sizeof(line), f)) {
        if (sscanf(line, "%1023s %lu %llu %llu", name, &in_init, &off, &sz)
            == 4) {
            g_devmap[name] = DevLoc{in_init != 0,
                                    static_cast<size_t>(off),
                                    static_cast<size_t>(sz)};
        }
    }
    fclose(f);
    log("loaded %zu device-global map entries from %s",
        g_devmap.size(), path);
}

// Resolve (and cache) the vanilla cubin's device address for a registered
// __device__ symbol via cudaGetSymbolAddress on its hostVar. Returns 0 if
// unknown.
CUdeviceptr vanilla_addr_for(const std::string& name) {
    // __managed__: the unified pointer is *hostVarPtrAddress, filled by
    // the runtime during registration. Deref fresh each call (not cached)
    // — it isn't known until after __cudaRegisterManagedVar completes.
    auto mit = g_name_to_managed_ptr.find(name);
    if (mit != g_name_to_managed_ptr.end() && mit->second) {
        return reinterpret_cast<CUdeviceptr>(*mit->second);
    }
    auto cit = g_vanilla_sym_addr.find(name);
    if (cit != g_vanilla_sym_addr.end()) return cit->second;
    CUdeviceptr addr = 0;
    auto hit = g_name_to_hostvar.find(name);
    if (hit != g_name_to_hostvar.end()) {
        void* a = nullptr;
        if (cudaGetSymbolAddress(&a, hit->second) == cudaSuccess) {
            addr = reinterpret_cast<CUdeviceptr>(a);
        }
    }
    g_vanilla_sym_addr[name] = addr;
    return addr;
}

// Sync every mapped __device__ global between the vanilla cubin's symbol
// and the lifted module's flat buffer. push_to_lifted=true copies
// vanilla->lifted (before a lifted launch); false copies lifted->vanilla
// (after). Keeps the host's direct (managed) reads, cudaMemcpyFromSymbol,
// and the lifted kernel's view of host writes all consistent.
void sync_device_globals(bool push_to_lifted) {
    if (!g_lifted_module || g_devmap.empty()) return;
    CUdeviceptr bss = 0, ini = 0;
    size_t nb = 0;
    bool have_bss =
        cuModuleGetGlobal(&bss, &nb, g_lifted_module,
                          "__slifter_nv_global") == CUDA_SUCCESS;
    bool have_ini =
        cuModuleGetGlobal(&ini, &nb, g_lifted_module,
                          "__slifter_nv_global_init") == CUDA_SUCCESS;
    for (auto& kv : g_devmap) {
        const DevLoc& d = kv.second;
        if (d.size == 0) continue;
        CUdeviceptr lifted_base = d.in_init ? ini : bss;
        if (d.in_init ? !have_ini : !have_bss) continue;
        CUdeviceptr van = vanilla_addr_for(kv.first);
        if (!van) continue;
        CUdeviceptr lifted = lifted_base + d.offset;
        CUresult r = push_to_lifted
            ? cuMemcpyDtoD(lifted, van, d.size)
            : cuMemcpyDtoD(van, lifted, d.size);
        if (r != CUDA_SUCCESS) {
            const char* msg = nullptr;
            cuGetErrorString(r, &msg);
            log("sync %s '%s' (%zu B) failed: %s",
                push_to_lifted ? "->lifted" : "->vanilla",
                kv.first.c_str(), d.size, msg ? msg : "?");
        }
    }
}

void ensure_init() {
    static bool tried = false;
    if (tried) return;
    tried = true;

    g_verbose = (getenv("PTXSWAP_VERBOSE") != nullptr);

    if (const char* ren = getenv("PTXSWAP_RENAME")) {
        std::string s(ren);
        size_t pos = 0;
        while (pos < s.size()) {
            size_t c = s.find(',', pos);
            if (c == std::string::npos) c = s.size();
            if (c > pos) g_rename_needles.emplace_back(s.substr(pos, c - pos));
            pos = c + 1;
        }
    }

    real_cudaLaunchKernel = reinterpret_cast<cuda_launch_t>(
        dlsym(RTLD_NEXT, "cudaLaunchKernel"));
    if (!real_cudaLaunchKernel) {
        fprintf(stderr, "[ptxswap] FATAL: dlsym(cudaLaunchKernel): %s\n",
                dlerror());
        return;
    }

    const char* ptx_path = getenv("PTXSWAP_PTX");
    if (!ptx_path) {
        log("PTXSWAP_PTX unset — passing through all launches");
        return;
    }

    // Force runtime to initialise its primary context, then borrow it
    // (instead of creating a fresh driver-API context which wouldn't
    // share allocations with the host app).
    cudaFree(nullptr);

    CUdevice dev;
    if (cuDeviceGet(&dev, 0) != CUDA_SUCCESS) return;
    CUcontext ctx;
    if (cuDevicePrimaryCtxRetain(&ctx, dev) != CUDA_SUCCESS) return;
    cuCtxSetCurrent(ctx);

    CUresult r = cuModuleLoad(&g_lifted_module, ptx_path);
    if (r != CUDA_SUCCESS) {
        const char* msg = nullptr;
        cuGetErrorString(r, &msg);
        fprintf(stderr, "[ptxswap] cuModuleLoad(%s) failed: %s\n",
                ptx_path, msg ? msg : "?");
        g_lifted_module = nullptr;
        return;
    }
    log("loaded lifted PTX from %s (%zu kernels recorded)",
        ptx_path, g_stub_to_name.size());

    // Resolve const_mem bank size from the lifted module (only needed
    // when a symbol map is present). const_mem is one flat global of
    // NUM_CONST_MEM_BANKS equal-sized banks; bank_size = bytes / banks.
    load_symmap();
    load_devmap();
    if (!g_symmap.empty()) {
        CUdeviceptr base = 0;
        size_t nbytes = 0;
        if (cuModuleGetGlobal(&base, &nbytes, g_lifted_module,
                              "const_mem") == CUDA_SUCCESS &&
            nbytes >= NUM_CONST_MEM_BANKS) {
            g_bank_size = nbytes / NUM_CONST_MEM_BANKS;
            log("const_mem: %zu bytes, bank_size=%zu", nbytes, g_bank_size);
        } else {
            log("const_mem global not found in lifted module — "
                "constant-symbol mirroring disabled");
        }
    }
}

// CUDA >= 12.3 can name a launch pointer directly. On CUDA 13 the
// pointer the device stub passes to __cudaLaunchKernel is NOT the
// hostFun that __cudaRegisterFunction recorded (H200 pilot 142531:
// 4 kernels registered, every launch arrived with an unknown pointer),
// so the stub map misses; cudaFuncGetName resolves those by NAME.
// On 12.2 the symbol is absent and the stub map alone suffices.
typedef cudaError_t (*func_get_name_t)(const char**, const void*);

const char* name_for_launch_ptr(const void* func) {
    static func_get_name_t get_name = []() {
        return reinterpret_cast<func_get_name_t>(
            dlsym(RTLD_NEXT, "cudaFuncGetName"));
    }();
    if (!get_name) return nullptr;
    const char* n = nullptr;
    if (get_name(&n, func) != cudaSuccess) return nullptr;
    return n;
}

// PTXSWAP_RENAME fallback: host kernel `name` is not in the lifted
// module; if it contains a configured needle, return the unique lifted
// kernel whose name contains the same needle (nullptr otherwise).
CUfunction fuzzy_resolve(const std::string& name) {
    if (g_rename_needles.empty()) return nullptr;
    for (const auto& needle : g_rename_needles) {
        if (name.find(needle) == std::string::npos) continue;
        unsigned count = 0;
        if (cuModuleGetFunctionCount(&count, g_lifted_module) !=
            CUDA_SUCCESS)
            return nullptr;
        std::vector<CUfunction> fns(count);
        if (cuModuleEnumerateFunctions(fns.data(), count,
                                       g_lifted_module) != CUDA_SUCCESS)
            return nullptr;
        CUfunction match = nullptr;
        const char* match_name = nullptr;
        int nmatch = 0;
        for (CUfunction f : fns) {
            const char* fn = nullptr;
            if (cuFuncGetName(&fn, f) != CUDA_SUCCESS || !fn) continue;
            if (std::string(fn).find(needle) != std::string::npos) {
                ++nmatch;
                match = f;
                match_name = fn;
            }
        }
        if (nmatch == 1) {
            log("RENAME '%s' -> lifted '%s' (needle '%s')", name.c_str(),
                match_name, needle.c_str());
            return match;
        }
        log("RENAME needle '%s' matches %d lifted kernels — not unique, "
            "skipping", needle.c_str(), nmatch);
    }
    return nullptr;
}

CUfunction resolve(const void* stub) {
    auto it = g_stub_cache.find(stub);
    if (it != g_stub_cache.end()) return it->second;

    std::string name;
    auto nit = g_stub_to_name.find(stub);
    if (nit != g_stub_to_name.end()) {
        name = nit->second;
    } else if (const char* n = name_for_launch_ptr(stub)) {
        log("stub %p unknown to the registration map; cudaFuncGetName "
            "says '%s'", stub, n);
        name = n;
    } else {
        log("no name recorded for stub %p — passing through", stub);
        g_stub_cache[stub] = nullptr;
        return nullptr;
    }

    CUfunction f = nullptr;
    CUresult r = cuModuleGetFunction(&f, g_lifted_module, name.c_str());
    if (r != CUDA_SUCCESS) {
        f = fuzzy_resolve(name);
        if (!f) {
            log("kernel %s not in lifted module — passing through",
                name.c_str());
            g_stub_cache[stub] = nullptr;
            return nullptr;
        }
        g_stub_cache[stub] = f;
        return f;
    }
    log("mapping stub %p -> lifted '%s'", stub, name.c_str());
    g_stub_cache[stub] = f;
    return f;
}

}  // namespace

// Interceptor: nvcc's host-side fatbin registration calls this at
// startup constructor time, once per kernel embedded in the original
// binary. We record the (stub -> mangled kernel name) mapping and
// forward to the real implementation so the original cubin still
// registers normally — fall-through paths can use it.
extern "C" void __cudaRegisterFunction(void** fatCubinHandle,
                                       const char* hostFun,
                                       char* deviceFun,
                                       const char* deviceName,
                                       int thread_limit,
                                       void* tid, void* bid,
                                       void* bDim, void* gDim,
                                       int* wSize) {
    if (!real_cudaRegisterFunction) {
        real_cudaRegisterFunction = reinterpret_cast<register_fn_t>(
            dlsym(RTLD_NEXT, "__cudaRegisterFunction"));
    }
    if (deviceFun && hostFun) {
        g_stub_to_name[reinterpret_cast<const void*>(hostFun)] = deviceFun;
    }
    if (real_cudaRegisterFunction) {
        real_cudaRegisterFunction(fatCubinHandle, hostFun, deviceFun,
                                  deviceName, thread_limit,
                                  tid, bid, bDim, gDim, wSize);
    }
}

// Interceptor: nvcc registers each __device__/__constant__ variable at
// startup via __cudaRegisterVar. We record (hostVar -> deviceName) so a
// later cudaMemcpyToSymbol(hostVar, ...) can be resolved to its device
// symbol name and mirrored into the lifted module's const_mem. Forward
// to the real implementation so the vanilla cubin still registers
// (fall-through kernels keep correct constant data).
extern "C" void __cudaRegisterVar(void** fatCubinHandle,
                                  char* hostVar,
                                  char* deviceAddress,
                                  const char* deviceName,
                                  int ext, size_t size,
                                  int constant, int global) {
    if (!real_cudaRegisterVar) {
        real_cudaRegisterVar = reinterpret_cast<register_var_t>(
            dlsym(RTLD_NEXT, "__cudaRegisterVar"));
    }
    if (hostVar && deviceName) {
        g_var_to_name[reinterpret_cast<const void*>(hostVar)] = deviceName;
        g_name_to_hostvar[deviceName] = reinterpret_cast<const void*>(hostVar);
    }
    if (real_cudaRegisterVar) {
        real_cudaRegisterVar(fatCubinHandle, hostVar, deviceAddress,
                             deviceName, ext, size, constant, global);
    }
}

// Interceptor: __managed__ globals are registered here, NOT via
// __cudaRegisterVar. hostVarPtrAddress is a void** whose pointee the
// runtime fills with the managed unified pointer the host reads/writes
// directly (hungarian-cuda's n_matches/zeros_size/goto_5/repeat_kernel).
// We record the void** so the launch-boundary device-global sync can
// deref it and keep the lifted buffer and the host's managed memory in
// step. Forward to the real implementation so the runtime still
// allocates the managed storage.
using register_managed_var_t = void (*)(void**, void**, char*,
                                        const char*, int, size_t,
                                        int, int);
register_managed_var_t real_cudaRegisterManagedVar = nullptr;
extern "C" void __cudaRegisterManagedVar(void** fatCubinHandle,
                                         void** hostVarPtrAddress,
                                         char* deviceAddress,
                                         const char* deviceName,
                                         int ext, size_t size,
                                         int constant, int global) {
    if (!real_cudaRegisterManagedVar) {
        real_cudaRegisterManagedVar =
            reinterpret_cast<register_managed_var_t>(
                dlsym(RTLD_NEXT, "__cudaRegisterManagedVar"));
    }
    if (real_cudaRegisterManagedVar) {
        real_cudaRegisterManagedVar(fatCubinHandle, hostVarPtrAddress,
                                    deviceAddress, deviceName, ext, size,
                                    constant, global);
    }
    if (hostVarPtrAddress && deviceName) {
        g_name_to_managed_ptr[deviceName] = hostVarPtrAddress;
    }
}

// Interceptor: cudaMemcpyToSymbol writes host data into a device
// __constant__/__device__ symbol. We ALWAYS perform the real copy first
// (preserves correctness for kernels that ride the vanilla cubin), then
// — if a lifted module is loaded AND a symbol map is present AND this
// symbol is a registered constant we know the (bank, offset) of —
// mirror the same bytes into the lifted module's flat `const_mem`
// global. When PTXSWAP_SYMMAP is unset, g_symmap is empty and this is a
// pure pass-through: no behavior change for any other test.
extern "C" cudaError_t cudaMemcpyToSymbol(const void* symbol,
                                          const void* src, size_t count,
                                          size_t offset,
                                          cudaMemcpyKind kind) {
    if (!real_cudaMemcpyToSymbol) {
        real_cudaMemcpyToSymbol = reinterpret_cast<memcpy_to_symbol_t>(
            dlsym(RTLD_NEXT, "cudaMemcpyToSymbol"));
    }
    cudaError_t rc = real_cudaMemcpyToSymbol
        ? real_cudaMemcpyToSymbol(symbol, src, count, offset, kind)
        : cudaErrorInitializationError;

    ensure_init();
    if (g_lifted_module && g_bank_size && !g_symmap.empty()) {
        auto nit = g_var_to_name.find(symbol);
        if (nit != g_var_to_name.end()) {
            auto sit = g_symmap.find(nit->second);
            if (sit != g_symmap.end()) {
                CUdeviceptr base = 0;
                size_t nbytes = 0;
                if (cuModuleGetGlobal(&base, &nbytes, g_lifted_module,
                                      "const_mem") == CUDA_SUCCESS) {
                    size_t byte = sit->second.bank * g_bank_size +
                                  sit->second.offset + offset;
                    CUresult r = cuMemcpyHtoD(base + byte, src, count);
                    if (r != CUDA_SUCCESS) {
                        const char* msg = nullptr;
                        cuGetErrorString(r, &msg);
                        log("cuMemcpyHtoD(const_mem+%zu, %zu) for '%s' "
                            "failed: %s", byte, count, nit->second.c_str(),
                            msg ? msg : "?");
                    } else {
                        log("mirrored %zu bytes of '%s' -> const_mem+%zu "
                            "(bank %u, off %zu)", count, nit->second.c_str(),
                            byte, sit->second.bank, sit->second.offset);
                    }
                }
            }
        }
    }
    return rc;
}

// The context's stack limit was sized when the VANILLA module loaded; a
// JIT-loaded lifted kernel with a bigger frame (per-thread Params staging
// can spill ~1-2KB vs the original's near-zero) silently overflows it —
// compute-sanitizer: "Out-of-range shared or local address" on frame STL —
// because the driver does NOT auto-grow the limit for driver-API modules.
// Raise CU_LIMIT_STACK_SIZE to the lifted function's local footprint before
// launch (reserve-only: cannot affect other kernels' correctness).
static void ensure_stack_limit(CUfunction lifted) {
    int local_sz = 0;
    CUresult qr = cuFuncGetAttribute(&local_sz,
                                     CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES,
                                     lifted);
    if (qr != CUDA_SUCCESS || local_sz <= 0) {
        log("stack-limit: attr query rc=%d local=%d", (int)qr, local_sz);
        return;
    }
    size_t cur = 0;
    if (cuCtxGetLimit(&cur, CU_LIMIT_STACK_SIZE) == CUDA_SUCCESS &&
        (size_t)local_sz > cur) {
        CUresult sr = cuCtxSetLimit(CU_LIMIT_STACK_SIZE, (size_t)local_sz);
        size_t after = 0;
        cuCtxGetLimit(&after, CU_LIMIT_STACK_SIZE);
        log("stack-limit: %zu -> %zu (need %d, set rc=%d)",
            cur, after, local_sz, (int)sr);
    }
}

// PTXSWAP_DUMP_PARAMS="needle:path:size" — on the first SWAPPED launch whose
// kernel name contains <needle>, write <size> bytes of the aggregate
// kernel-arg struct (args[0], the cutlass-style Params) to <path>, then
// disarm. Used to regenerate the captured Params blobs that the CPU-backend
// FA drivers replay (benchmark/flash_attention/params/).
static void maybe_dump_params(const void* stub, void** args) {
    static bool done = false;
    if (done) return;
    const char* spec = getenv("PTXSWAP_DUMP_PARAMS");
    if (!spec || !args || !args[0]) return;
    const char* c1 = strchr(spec, ':');
    const char* c2 = c1 ? strrchr(spec, ':') : nullptr;
    if (!c1 || c2 == c1) return;                     // need needle:path:size
    std::string needle(spec, c1 - spec);
    std::string path(c1 + 1, c2 - c1 - 1);
    size_t n = strtoul(c2 + 1, nullptr, 0);
    auto it = g_stub_to_name.find(stub);
    if (it == g_stub_to_name.end() ||
        it->second.find(needle) == std::string::npos) return;
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) { fprintf(stderr, "[ptxswap] dump-params: cannot open %s\n",
                      path.c_str()); return; }
    fwrite(args[0], 1, n, f);
    fclose(f);
    fprintf(stderr, "[ptxswap] dumped %zu param bytes of %s -> %s\n",
            n, it->second.c_str(), path.c_str());
    done = true;
}

// Callers run ensure_init() BEFORE reading the real-fn global they pass
// in — on the very first launch the global is only populated by init.
static cudaError_t launch_common(cuda_launch_t fallthrough,
                                 const void* func,
                                 dim3 gridDim, dim3 blockDim,
                                 void** args, size_t sharedMem,
                                 cudaStream_t stream) {
    if (!fallthrough) {
        return cudaErrorInitializationError;
    }
    if (!g_lifted_module) {
        return fallthrough(func, gridDim, blockDim,
                           args, sharedMem, stream);
    }

    CUfunction lifted = resolve(func);
    if (!lifted) {
        return fallthrough(func, gridDim, blockDim,
                           args, sharedMem, stream);
    }

    // The device-global sync issues explicit cuStreamSynchronize +
    // cuMemcpyDtoD, which are ILLEGAL while the stream is capturing a CUDA
    // graph (graphExecution). Skip the sync entirely during capture — the
    // kernel is still captured into the graph, and capture-based tests
    // don't share __device__ globals with the host through this path.
    CUstreamCaptureStatus cap = CU_STREAM_CAPTURE_STATUS_NONE;
    bool capturing = false;
    if (!g_devmap.empty() &&
        cuStreamIsCapturing(reinterpret_cast<CUstream>(stream), &cap)
            == CUDA_SUCCESS) {
        capturing = (cap != CU_STREAM_CAPTURE_STATUS_NONE);
    }

    // Mirror host-visible __device__ globals into the lifted module's flat
    // buffer before the kernel runs (sees host writes + prior managed
    // updates). Serialise first so pending host/symbol copies have landed.
    if (!g_devmap.empty() && !capturing) {
        cuStreamSynchronize(reinterpret_cast<CUstream>(stream));
        sync_device_globals(/*push_to_lifted=*/true);
    }

    // >48KB dynamic shared needs the opt-in attribute raised on the LIFTED
    // function — the app raised it on the ORIGINAL stub only (same reason as
    // the ExC hook below; FA3's non-cluster dispatch launches ~216KB dynamic
    // through this plain path and otherwise fails with INVALID_VALUE).
    if (sharedMem > 48 * 1024) {
        cuFuncSetAttribute(lifted,
                           CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                           (int)sharedMem);
    }
    ensure_stack_limit(lifted);
    maybe_dump_params(func, args);
    CUresult r = cuLaunchKernel(
        lifted,
        gridDim.x, gridDim.y, gridDim.z,
        blockDim.x, blockDim.y, blockDim.z,
        sharedMem, reinterpret_cast<CUstream>(stream),
        args, /*extra=*/nullptr);
    if (r != CUDA_SUCCESS) {
        const char* msg = nullptr;
        cuGetErrorString(r, &msg);
        fprintf(stderr, "[ptxswap] cuLaunchKernel failed: %s\n",
                msg ? msg : "?");
        return cudaErrorLaunchFailure;
    }

    // Copy results back so the host's managed reads / cudaMemcpyFromSymbol
    // see what the lifted kernel produced.
    if (!g_devmap.empty() && !capturing) {
        cuStreamSynchronize(reinterpret_cast<CUstream>(stream));
        sync_device_globals(/*push_to_lifted=*/false);
    }
    return cudaSuccess;
}

// FA3 / cluster kernels launch through cudaLaunchKernelExC (launch config +
// attributes, e.g. CLUSTER_DIMENSION). Without this hook every ExC launch
// silently rides the vanilla cubin (fa_host pilot: zero 'mapping stub').
// Translate the runtime config to a driver CUlaunchConfig (attribute ids and
// value unions are layout-compatible) and dispatch the lifted CUfunction via
// cuLaunchKernelEx. Kernels needing >48KB dynamic smem also need the driver
// MAX_DYNAMIC_SHARED_SIZE_BYTES attribute raised on the lifted function first
// (the app raised it on the ORIGINAL stub only).
typedef cudaError_t (*cuda_launch_exc_t)(const cudaLaunchConfig_t*,
                                         const void*, void**);
static cuda_launch_exc_t real_cudaLaunchKernelExC = nullptr;

extern "C" cudaError_t cudaLaunchKernelExC(const cudaLaunchConfig_t* cfg,
                                           const void* func, void** args) {
    ensure_init();
    if (!real_cudaLaunchKernelExC)
        real_cudaLaunchKernelExC = reinterpret_cast<cuda_launch_exc_t>(
            dlsym(RTLD_NEXT, "cudaLaunchKernelExC"));
    if (!g_lifted_module || !cfg || !real_cudaLaunchKernelExC)
        return real_cudaLaunchKernelExC
                   ? real_cudaLaunchKernelExC(cfg, func, args)
                   : cudaErrorInitializationError;
    CUfunction lifted = resolve(func);
    if (!lifted)
        return real_cudaLaunchKernelExC(cfg, func, args);

    if (cfg->dynamicSmemBytes > 48 * 1024) {
        cuFuncSetAttribute(lifted,
                           CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                           (int)cfg->dynamicSmemBytes);
    }
    ensure_stack_limit(lifted);
    maybe_dump_params(func, args);
    CUlaunchAttribute attrs[16];
    unsigned n = cfg->numAttrs > 16 ? 16 : cfg->numAttrs;
    for (unsigned i = 0; i < n; i++) {
        attrs[i].id = (CUlaunchAttributeID)cfg->attrs[i].id;
        memcpy(&attrs[i].value, &cfg->attrs[i].val, sizeof(attrs[i].value));
    }
    CUlaunchConfig lc = {};
    lc.gridDimX = cfg->gridDim.x; lc.gridDimY = cfg->gridDim.y;
    lc.gridDimZ = cfg->gridDim.z;
    lc.blockDimX = cfg->blockDim.x; lc.blockDimY = cfg->blockDim.y;
    lc.blockDimZ = cfg->blockDim.z;
    lc.sharedMemBytes = (unsigned)cfg->dynamicSmemBytes;
    lc.hStream = reinterpret_cast<CUstream>(cfg->stream);
    lc.attrs = attrs; lc.numAttrs = n;
    CUresult r = cuLaunchKernelEx(&lc, lifted, args, /*extra=*/nullptr);
    if (r != CUDA_SUCCESS) {
        const char* msg = nullptr;
        cuGetErrorString(r, &msg);
        fprintf(stderr, "[ptxswap] cuLaunchKernelEx failed: %s\n",
                msg ? msg : "?");
        return cudaErrorLaunchFailure;
    }
    return cudaSuccess;
}

extern "C" cudaError_t cudaLaunchKernel(const void* func,
                                        dim3 gridDim, dim3 blockDim,
                                        void** args, size_t sharedMem,
                                        cudaStream_t stream) {
    ensure_init();
    return launch_common(real_cudaLaunchKernel, func, gridDim, blockDim,
                         args, sharedMem, stream);
}

// CUDA 13 device stubs launch through the INTERNAL entry points
// __cudaLaunchKernel / __cudaLaunchKernel_ptsz instead of the public
// cudaLaunchKernel — without these hooks a 13.x-built app silently rides
// the vanilla cubin with zero mapping-stub records. Same signature as the
// public symbol; forward to
// the real internal entry (RTLD_NEXT unversioned lookup returns the
// default-versioned cudart definition).
static cuda_launch_t real_cudaLaunchKernel_internal = nullptr;
static cuda_launch_t real_cudaLaunchKernel_ptsz = nullptr;

extern "C" cudaError_t __cudaLaunchKernel(const void* func,
                                          dim3 gridDim, dim3 blockDim,
                                          void** args, size_t sharedMem,
                                          cudaStream_t stream) {
    ensure_init();
    if (!real_cudaLaunchKernel_internal) {
        real_cudaLaunchKernel_internal = reinterpret_cast<cuda_launch_t>(
            dlsym(RTLD_NEXT, "__cudaLaunchKernel"));
        if (!real_cudaLaunchKernel_internal)
            real_cudaLaunchKernel_internal = real_cudaLaunchKernel;
    }
    return launch_common(real_cudaLaunchKernel_internal, func, gridDim,
                         blockDim, args, sharedMem, stream);
}

extern "C" cudaError_t __cudaLaunchKernel_ptsz(const void* func,
                                               dim3 gridDim, dim3 blockDim,
                                               void** args, size_t sharedMem,
                                               cudaStream_t stream) {
    ensure_init();
    if (!real_cudaLaunchKernel_ptsz) {
        real_cudaLaunchKernel_ptsz = reinterpret_cast<cuda_launch_t>(
            dlsym(RTLD_NEXT, "__cudaLaunchKernel_ptsz"));
        if (!real_cudaLaunchKernel_ptsz)
            real_cudaLaunchKernel_ptsz = real_cudaLaunchKernel;
    }
    return launch_common(real_cudaLaunchKernel_ptsz, func, gridDim,
                         blockDim, args, sharedMem, stream);
}
