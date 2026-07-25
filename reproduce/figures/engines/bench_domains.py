#!/usr/bin/env python3
"""Best-effort domain classification for HecBench and other benchmarks.

Maps benchmark name → domain category for paper figures.
Update this file as needed — the mapping is approximate and should be reviewed.

Usage:
    from reproduce.figures.engines.bench_domains import get_domain, DOMAINS
    domain = get_domain("softmax-cuda")  # → "ML/DNN"
"""

# 10 domain categories for paper figures
DOMAINS = [
    "Linear Algebra",
    "ML/DNN",
    "Simulation",
    "Graph",
    "Image/Vision",
    "Crypto",
    "Science",
    "Sort/Search",
    "Stencil/PDE",
    "Infrastructure",
]

# Keyword-based classification rules (checked in order, first match wins)
# Each rule: (domain, [keywords that trigger match])
_RULES = [
    # ML/DNN — neural network ops, attention, normalization, quantization
    ("ML/DNN", [
        "attention", "softmax", "layernorm", "batchnorm", "bn-", "dropout",
        "relu", "gelu", "geglu", "glu", "swish", "sigmoid",
        "backprop", "adam", "adamw", "sgd", "lr-", "lrn",
        "crossEntropy", "nlll", "scel", "logprob", "perplexity",
        "mnist", "resnet", "word2vec", "tsne", "dense-embedding",
        "addBias", "channelShuffle", "channelSum", "rotary",
        "rowwiseMoments", "moe", "qkv", "addBiasQKV",
        "quant3MatMul", "quantAQLM", "quantBnB", "quantVLLM",
        "lut-gemm", "wmma", "tensorAccessor", "tensorT",
        "dwconv", "convolution", "pool", "maxpool3d",
        "face", "nms", "knn", "kmeans",
    ]),
    # Linear Algebra — BLAS, sparse, dense matrix
    ("Linear Algebra", [
        "gemm", "gemv", "spmm", "spmv", "geam", "axpy", "saxpy",
        "blas-", "chemv", "lud", "ludb", "gels", "hpl",
        "slu", "tridiagonal", "bicgstab", "jacobi",
        "ge-spmm", "spgemm", "spgeam", "simpleSpmv",
        "sddmm", "mmcsf", "norm2", "svd3x3",
        "matrixT", "matrix-rotate", "determinant",
        "lanczos", "eigenvalue",
        "CB_", "CD_",  # cublas/cudnn prefixed
    ]),
    # Stencil/PDE — structured grid, finite difference, heat/wave
    ("Stencil/PDE", [
        "stencil", "heat", "hotspot", "fdtd", "laplace", "iso2dfd",
        "jacobi", "rtm8", "axhelm", "minimod", "hypterm", "dslash",
        "cfd", "burger", "lid-driven", "d2q9", "d3q19", "lulesh",
        "rayleighBenard", "miniWeather", "easyWave", "sw4ck",
        "wsm5", "thomas",
    ]),
    # Simulation — particles, molecular, monte carlo, physics
    ("Simulation", [
        "nbody", "lavaMD", "particle", "sph",
        "fluidSim", "ising", "cobahh", "lif", "langevin",
        "metropolis", "montecarlo", "feynman", "gibbs",
        "collision", "projectile", "bonds", "libor",
        "black-scholes", "binomial", "minibude", "nbnxm",
        "heartwall", "myocyte", "testSNAP", "su3",
        "rushlarsen", "hexciton", "sheath", "goulash",
        "s3d", "haccmk", "simplemoc", "minisweep",
        "damage", "vmc", "egs", "cooling", "snake",
        "md-cuda", "mdh", "mcmd", "mcpr", "pso",
        "reaction", "multimaterial", "flame", "slit",
        "rainflow", "grrt", "kmc", "ga-",
        "colombar", "bh-",
    ]),
    # Graph — graph algorithms, trees
    ("Graph", [
        "bfs", "sssp", "page-rank", "cc-", "ccl",
        "b+tree", "btree", "graphB", "graphExecution",
        "mis", "jaccard", "floydwarshall", "wyllie",
        "streamcluster", "hbc", "tsp", "nqueen",
        "pathfinder", "colombar", "snicit",
    ]),
    # Image/Vision — image processing, filters
    ("Image/Vision", [
        "bilateral", "boxfilter", "sobel", "gaussian",
        "debayer", "dct8x8", "dxtc2", "filter",
        "gamma-correction", "histogram", "mandelbrot",
        "medianfilter", "morphology", "opticalFlow",
        "recursiveGaussian", "resize", "seam-carving",
        "tonemapping", "background-subtract", "aobench",
        "bezier-surface", "colorwheel", "depixel",
        "distort", "gabor", "hogbom", "voxelization",
        "zoom", "ssim", "bm3d", "marchingCubes",
        "surfel", "pnpoly", "doh", "ced",
        "flip", "srad", "perlin",
    ]),
    # Crypto — cryptography, hashing, security
    ("Crypto", [
        "aes", "chacha20", "ecdh", "secp256k1",
        "md5hash", "murmurhash", "jenkins-hash",
        "keccak", "crc64", "bitcracker", "merkle",
        "present",
    ]),
    # Science — bioinformatics, geoscience, chemistry, physics apps
    ("Science", [
        "leukocyte", "dna", "prna", "frna", "phmm",
        "epistasis", "tissue", "bsw", "minimap2",
        "nw-cuda", "extend2", "deredundancy",
        "lombscargle", "tpacf", "geodesic",
        "vanGenuchten", "eikonal", "clenergy",
        "xlqc", "ccsd", "rsbench", "xsbench",
        "zmddft", "degrid", "fhd", "mriQ",
        "gpp", "che-", "diamond", "henry",
        "s8n", "lci", "daphne",
        "amgmk", "miniFE", "miniDGS",
        "expdist", "fresnel", "entropy",
        "matern", "kurtosis", "hellinger",
        "chi2", "haversine", "hausdorff",
        "lebesgue", "michalewicz", "romberg",
        "aidw", "all-pairs-distance",
        "asmooth", "asta", "adv",
        "halo-finder", "qtclustering",
        "cm-cuda", "car-", "ace-",
        "ccs-", "clink", "cmp-",
        "cbsfil", "crs-", "dp-cuda", "dpid",
        "logan", "log2-", "pcc",
        "wlcpow", "winograd",
        "hmm-cuda", "gru-", "kalman",
        "ldpc", "ntt-",
        "rsc", "rsmt",
        "lsqt", "hungarian",
        "si-cuda", "is-cuda", "ne-cuda",
        "fft",  "fwt", "hwt1d",
        "bmf", "gmm",
        "spaxpby", "spd2s", "spm-cuda", "spnnz",
        "sps2d", "spsm", "sptrsv",
        "mr-cuda", "mrc", "mpc",
        "bn-cuda",
        "qrg", "qem", "kiss",
        "nn-cuda", "langford", "lda",
        "compute-score", "score-",
        "sc-cuda", "ss-cuda", "sss-",
        "stsg", "tgvnn", "tqs", "tsa",
        "local-ht", "logic-resim", "logic-rewrite",
        "pns", "pointwise", "rfs",
        "lr-cuda", "gd-",
        "ddbp", "rodrigues",
        "bspline", "inversek2j",
        "overlay", "remap",
        "mask-cuda", "aop", "affine",
        "cross-cuda", "contract",
        "sosfil", "sparkler", "wedford",
        "zeropoint", "vol2col", "unfold",
        "mt-cuda", "mtf",
        "cc-cuda", "gerbil",
        "sa-cuda", "sad-",
        "columnaSolver",
    ]),
    # Sort/Search — sorting, searching, selection
    ("Sort/Search", [
        "sort", "radixsort", "quicksort", "bitonic-sort",
        "segsort", "hybridsort", "warpsort",
        "bsearch", "grep", "linearprobing",
        "merge-", "scan", "scan2", "scan3",
        "histogram", "bincount", "segment-reduce",
        "reduce", "prefix",
    ]),
    # Infrastructure — language features, memory, bandwidth, RNG, compression, atomics
    ("Infrastructure", [
        "atomic", "malloc", "memcpy", "memtest",
        "babelstream", "triad", "mixbench", "maxFlops",
        "shmembench", "cmembench", "blockAccess",
        "kernelLaunch", "concurrentKernels", "dispatch",
        "stream", "prefetch", "zerocopy", "overlap",
        "pingpong", "p2p", "nosync", "threadfence",
        "vote", "shuffle", "warpexchange", "blockexchange",
        "assert", "clock", "divergence",
        "sobol", "mrg32k3a", "rng-wallace", "lfib4", "urng",
        "lzss", "bwt", "rle", "fpc", "fpdc", "ans-",
        "bitpacking", "bitpermute", "popcount",
        "interleave", "concat", "pad-", "split",
        "permute", "permutate", "layout", "pitch",
        "conversion", "intrinsics", "complex",
        "openmp", "allreduce", "ring-",
        "aligned-types", "simpleMultiDevice",
        "loopback", "match-cuda", "nonzero",
        "sampling", "multinomial",
        "f16atomic", "f16max", "f16sp",
        "scatter", "scatterAdd",
        "randomAccess", "wordcount", "idivide",
        "accuracy", "stddev", "minmax",
        "reverse-cuda", "reverse2D",
        "p4-", "interval", "ert-",
        "adjacent", "coordinates",
    ]),
]


# Ground-truth overrides from previous paper's benchmark→category mapping.
# Maps: Math→Linear Algebra, Computer vision→Image/Vision, Machine learning→ML/DNN,
#        Language and kernel features→Infrastructure, Bandwidth→Infrastructure,
#        Data compression and reduction→Infrastructure, Sorting→Sort/Search,
#        Search→Sort/Search, Graph and Tree→Graph, Bioinformatics→Science,
#        Geoscience→Science, Cryptography→Crypto, Data enc/dec→Infrastructure,
#        RNG→Infrastructure, etc→best-fit.
_EXACT = {
    # --- From previous paper's verified mapping ---
    # Math → Linear Algebra
    "blas-gemmEx-cuda": "Linear Algebra", "blas-dot-cuda": "Linear Algebra",
    "blas-fp8gemm-cuda": "Linear Algebra", "blas-gemmEx2-cuda": "Linear Algebra",
    "blas-gemmStridedBatched-cuda": "Linear Algebra", "blas-gemmBatched-cuda": "Linear Algebra",
    "cross-cuda": "Linear Algebra", "f16max-cuda": "Linear Algebra",
    "geam-cuda": "Linear Algebra", "gels-cuda": "Linear Algebra",
    "jacobi-cuda": "Linear Algebra", "lanczos-cuda": "Linear Algebra",
    "ludb-cuda": "Linear Algebra", "matrix-rotate-cuda": "Linear Algebra",
    "mrg32k3a-cuda": "Linear Algebra", "norm2-cuda": "Linear Algebra",
    "nqueen-cuda": "Linear Algebra", "phmm-cuda": "Linear Algebra",
    "reverse2D-cuda": "Linear Algebra", "sddmm-batch-cuda": "Linear Algebra",
    "spgemm-cuda": "Linear Algebra", "spmm-cuda": "Linear Algebra",
    "sps2d-cuda": "Linear Algebra", "spnnz-cuda": "Linear Algebra",
    "spd2s-cuda": "Linear Algebra", "wyllie-cuda": "Linear Algebra",
    "chemv-cuda": "Simulation",  # user classified as Simulation, not Linear Algebra
    # Computer vision and image processing → Image/Vision
    "affine-cuda": "Image/Vision", "cbsfil-cuda": "Image/Vision",
    "convolutionSeperable-cuda": "Image/Vision", "convolution3D-cuda": "Image/Vision",
    "dct8x8-cuda": "Image/Vision", "face-cuda": "Image/Vision",
    "medianfilter-cuda": "Image/Vision", "recursiveGaussian-cuda": "Image/Vision",
    "stencil3d-cuda": "Image/Vision",  # user classified as vision, not Stencil
    # Machine learning → ML/DNN
    "accuracy-cuda": "ML/DNN", "backprop-cuda": "ML/DNN",
    "channelSum-cuda": "ML/DNN", "channelShuffle-cuda": "ML/DNN",
    "concat-cuda": "ML/DNN", "dense-embedding-cuda": "ML/DNN",
    "dwconv-cuda": "ML/DNN", "mask-cuda": "ML/DNN",
    "mrc-cuda": "ML/DNN", "permute-cuda": "ML/DNN",
    "rotary-cuda": "ML/DNN", "streamcluster-cuda": "ML/DNN",
    "winograd-cuda": "ML/DNN",
    # Simulation → Simulation
    "bspline-vgh-cuda": "Simulation", "fdtd3d-cuda": "Simulation",
    "heat2d-cuda": "Simulation", "iso2dfd-cuda": "Simulation",
    "laplace3d-cuda": "Simulation", "pathfinder-cuda": "Simulation",
    "reaction-cuda": "Simulation", "rtm8-cuda": "Simulation",
    "slit-cuda": "Simulation", "tissue-cuda": "Simulation",
    "tqs-cuda": "Simulation",
    # Graph and Tree → Graph
    "floydwarshall-cuda": "Graph", "sssp-cuda": "Graph",
    # Sort/Search (Sorting + Search)
    "bitonic-sort-cuda": "Sort/Search", "bsearch-cuda": "Sort/Search",
    "radixsort-cuda": "Sort/Search", "split-cuda": "Sort/Search",
    "bfs-cuda": "Sort/Search", "ss-cuda": "Sort/Search",
    # Cryptography → Crypto
    "ecdh-cuda": "Crypto", "keccaktreehash-cuda": "Crypto",
    # Bioinformatics / Geoscience → Science
    "all-pairs-distance-cuda": "Science", "hausdorff-cuda": "Science",
    "nw-cuda": "Science",
    # Language and kernel features → Infrastructure
    "aligned-types-cuda": "Infrastructure", "asta-cuda": "Infrastructure",
    "atomicAggregate-cuda": "Infrastructure", "atomicIntrinsics-cuda": "Infrastructure",
    "blockAccess-cuda": "Infrastructure", "blockexchange-cuda": "Infrastructure",
    "clock-cuda": "Infrastructure", "dispatch-cuda": "Infrastructure",
    "interleave-cuda": "Infrastructure", "kernelLaunch-cuda": "Infrastructure",
    "layout-cuda": "Infrastructure", "mallocFree-cuda": "Infrastructure",
    "overlap-cuda": "Infrastructure", "p2p-cuda": "Infrastructure",
    "popcount-cuda": "Infrastructure", "prefetch-cuda": "Infrastructure",
    "reverse-cuda": "Infrastructure", "ring-cuda": "Infrastructure",
    "shuffle-cuda": "Infrastructure", "simpleMultiDevice-cuda": "Infrastructure",
    "streamCreateCopyDestroy-cuda": "Infrastructure",
    "streamOrderedAllocation-cuda": "Infrastructure",
    "streamPriority-cuda": "Infrastructure", "streamUM-cuda": "Infrastructure",
    "tensorAccessor-cuda": "Infrastructure", "threadfence-cuda": "Infrastructure",
    "warpexchange-cuda": "Infrastructure", "wmma-cuda": "Infrastructure",
    "zerocopy-cuda": "Infrastructure",
    # Bandwidth → Infrastructure
    "babelstream-cuda": "Infrastructure", "memcpy-cuda": "Infrastructure",
    "memtest-cuda": "Infrastructure", "shmembench-cuda": "Infrastructure",
    "triad-cuda": "Infrastructure",
    # Data compression and reduction → Infrastructure
    "bscan-cuda": "Infrastructure", "bwt-cuda": "Infrastructure",
    "filter-cuda": "Infrastructure", "fpc-cuda": "Infrastructure",
    "scan2-cuda": "Infrastructure",
    # Data encoding/decoding → Infrastructure
    "crs-cuda": "Infrastructure",
    # RNG → Infrastructure
    "rng-wallace-cuda": "Infrastructure",
    # "etc" → best-fit categories
    "bicgstab-cuda": "Linear Algebra", "linearprobing-cuda": "Sort/Search",
    "adjacent-cuda": "Infrastructure", "spaxpby-cuda": "Linear Algebra",
    "spsm-cuda": "Linear Algebra",
    # --- Additional short/ambiguous name overrides ---
    "ace-cuda": "Science", "ans-cuda": "Infrastructure", "atan2-cuda": "Infrastructure",
    "bh-cuda": "Simulation", "bn-cuda": "ML/DNN", "car-cuda": "Simulation",
    "cc-cuda": "Graph", "ccs-cuda": "Linear Algebra", "che-cuda": "Science",
    "cm-cuda": "Science", "cmp-cuda": "Infrastructure", "columnarSolver-cuda": "Linear Algebra",
    "dp-cuda": "Science", "ert-cuda": "Infrastructure", "extrema-cuda": "Sort/Search",
    "frechet-cuda": "Science", "fsm-cuda": "Science", "ga-cuda": "Simulation",
    "gc-cuda": "Infrastructure", "gd-cuda": "Science", "gru-cuda": "ML/DNN",
    "hmm-cuda": "ML/DNN", "is-cuda": "Sort/Search", "keogh-cuda": "Science",
    "log2-cuda": "Infrastructure", "lr-cuda": "ML/DNN", "match-cuda": "Infrastructure",
    "md-cuda": "Simulation", "meanshift-cuda": "ML/DNN", "merge-cuda": "Sort/Search",
    "minkowski-cuda": "Science", "mr-cuda": "Science", "mt-cuda": "Infrastructure",
    "ne-cuda": "Science", "nn-cuda": "ML/DNN", "ntt-cuda": "Crypto",
    "p4-cuda": "Infrastructure", "pad-cuda": "Infrastructure",
    "sa-cuda": "Simulation", "sad-cuda": "Image/Vision", "sc-cuda": "Science",
    "score-cuda": "Science", "si-cuda": "Science", "spm-cuda": "Linear Algebra",
    "sss-cuda": "Science",
}


# Prefix-based classification for SDK/ML/cutlass suites
# Names from collect_eval_stats.py come as "cublas/kernelname", "cudnn_cnn/kernelname", etc.
_PREFIX_DOMAINS = {
    "cublas/": "Linear Algebra",
    "cublaslt/": "Linear Algebra",
    "cudnn_cnn/": "ML/DNN",
    "cudnn_ops/": "ML/DNN",
    "cudnn_adv/": "ML/DNN",
    "tensorop_": "ML/DNN",      # test/ml/ tensor-core ops
    "cutlass/": "ML/DNN",       # CUTLASS examples (GEMM, conv, attention)
    "flash_": "ML/DNN",         # Flash Attention kernels
}


def get_domain(bench_name: str) -> str:
    """Classify a benchmark name into a domain category.

    Args:
        bench_name: benchmark directory name, e.g. "softmax-cuda", "cublas/sgemm_nn"

    Returns:
        Domain string from DOMAINS list, or "Other" if unclassified.
    """
    # Check exact overrides first
    if bench_name in _EXACT:
        return _EXACT[bench_name]

    # Check prefix-based (SDK/ML/cutlass suites)
    for prefix, domain in _PREFIX_DOMAINS.items():
        if bench_name.startswith(prefix):
            return domain

    # Strip -cuda suffix
    name = bench_name.replace("-cuda", "")

    for domain, keywords in _RULES:
        for kw in keywords:
            if kw.lower() in name.lower():
                return domain
    return "Other"


def classify_all(bench_names):
    """Classify a list of benchmark names. Returns dict: name → domain."""
    return {name: get_domain(name) for name in bench_names}


def print_summary(classification):
    """Print domain distribution summary."""
    from collections import Counter
    counts = Counter(classification.values())
    print(f"\n{'Domain':<20s} {'Count':>5s}")
    print("-" * 27)
    for domain in DOMAINS + ["Other"]:
        if domain in counts:
            print(f"{domain:<20s} {counts[domain]:>5d}")
    print("-" * 27)
    print(f"{'Total':<20s} {sum(counts.values()):>5d}")


if __name__ == "__main__":
    import sys
    from pathlib import Path

    # Default: classify all hecbench
    hecbench_dir = Path(__file__).resolve().parent.parent.parent / "test" / "hecbench"
    names = sorted([d.name for d in hecbench_dir.iterdir() if d.is_dir()])
    classification = classify_all(names)
    print_summary(classification)

    # Print "Other" for manual review
    others = [n for n, d in classification.items() if d == "Other"]
    if others:
        print(f"\nUnclassified ({len(others)}):")
        for n in others:
            print(f"  {n}")
