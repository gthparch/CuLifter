#include <iostream>
#include <cuda_runtime.h>
#include <mma.h>
#include <cstdlib>
#include <cmath>
#include <iomanip>
#include <chrono>

using namespace nvcuda;

// WMMA dimensions - for half precision on Volta/Turing/Ampere
const int WMMA_M = 16;
const int WMMA_N = 16;
const int WMMA_K = 16;

// Matrix dimensions (must be multiples of WMMA tile dimensions)
const int M = 256;  // Rows of A and C
const int N = 256;  // Cols of B and C
const int K = 256;  // Cols of A and rows of B

// Kernel using WMMA for matrix multiplication C = A * B
__global__ void wmma_gemm(half *a, half *b, float *c, int M, int N, int K) {
    // Tile dimensions
    const int warpM = (M + WMMA_M - 1) / WMMA_M;
    const int warpN = (N + WMMA_N - 1) / WMMA_N;
    
    // Warp and lane identification
    const int warpId = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    const int laneId = threadIdx.x % 32;
    
    // Compute warp's position in output matrix
    const int warpRow = warpId / warpN;
    const int warpCol = warpId % warpN;
    
    // Skip if this warp is outside the matrix bounds
    if (warpRow >= warpM || warpCol >= warpN) return;
    
    // Declare fragments for matrix multiplication
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
    
    // Initialize accumulator fragment to zero
    wmma::fill_fragment(c_frag, 0.0f);
    
    // Loop over K dimension
    for (int k = 0; k < K; k += WMMA_K) {
        // Calculate matrix addresses for this tile
        half *aPtr = a + warpRow * WMMA_M * K + k;
        half *bPtr = b + k * N + warpCol * WMMA_N;
        
        // Load fragments
        wmma::load_matrix_sync(a_frag, aPtr, K);
        wmma::load_matrix_sync(b_frag, bPtr, N);
        
        // Perform matrix multiplication
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }
    
    // Store result
    float *cPtr = c + warpRow * WMMA_M * N + warpCol * WMMA_N;
    wmma::store_matrix_sync(cPtr, c_frag, N, wmma::mem_row_major);
}

// CPU matrix multiplication for verification
void cpu_gemm(float *a, float *b, float *c, int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += a[i * K + k] * b[k * N + j];
            }
            c[i * N + j] = sum;
        }
    }
}

// Convert float array to half precision
void float_to_half(float *input, half *output, int size) {
    for (int i = 0; i < size; i++) {
        output[i] = __float2half(input[i]);
    }
}

// Convert half precision array to float
void half_to_float(half *input, float *output, int size) {
    for (int i = 0; i < size; i++) {
        output[i] = __half2float(input[i]);
    }
}

// Initialize matrix with random values
void init_matrix(float *mat, int size) {
    for (int i = 0; i < size; i++) {
        mat[i] = (float)(rand() % 10) / 10.0f;  // Random values between 0 and 1
    }
}

// Check if results match within tolerance
bool check_results(float *gpu, float *cpu, int size) {
    const float tolerance = 1e-2f;  // Relaxed tolerance for half precision
    bool correct = true;
    int mismatches = 0;
    
    for (int i = 0; i < size; i++) {
        float diff = fabs(gpu[i] - cpu[i]);
        if (diff > tolerance) {
            if (mismatches < 10) {  // Print first 10 mismatches
                std::cout << "Mismatch at index " << i << ": GPU = " << gpu[i] 
                         << ", CPU = " << cpu[i] << ", diff = " << diff << std::endl;
            }
            mismatches++;
            correct = false;
        }
    }
    
    if (mismatches > 0) {
        std::cout << "Total mismatches: " << mismatches << " out of " << size << std::endl;
    }
    
    return correct;
}

int main() {
    // Check for Tensor Core support
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    if (prop.major < 7) {
        std::cerr << "Error: Tensor Cores require compute capability 7.0 or higher" << std::endl;
        std::cerr << "Your device has compute capability " << prop.major << "." << prop.minor << std::endl;
        return 1;
    }
    
    std::cout << "Running on: " << prop.name << " (compute capability " 
              << prop.major << "." << prop.minor << ")" << std::endl;
    
    // Allocate host memory
    float *h_A = new float[M * K];
    float *h_B = new float[K * N];
    float *h_C_gpu = new float[M * N];
    float *h_C_cpu = new float[M * N];
    
    // Initialize matrices
    std::cout << "\nInitializing matrices..." << std::endl;
    std::cout << "Matrix A: " << M << " x " << K << std::endl;
    std::cout << "Matrix B: " << K << " x " << N << std::endl;
    std::cout << "Matrix C: " << M << " x " << N << std::endl;
    
    init_matrix(h_A, M * K);
    init_matrix(h_B, K * N);
    
    // Allocate device memory
    half *d_A, *d_B;
    float *d_C;
    
    cudaMalloc(&d_A, M * K * sizeof(half));
    cudaMalloc(&d_B, K * N * sizeof(half));
    cudaMalloc(&d_C, M * N * sizeof(float));
    
    // Convert to half precision and copy to device
    half *h_A_half = new half[M * K];
    half *h_B_half = new half[K * N];
    
    float_to_half(h_A, h_A_half, M * K);
    float_to_half(h_B, h_B_half, K * N);
    
    cudaMemcpy(d_A, h_A_half, M * K * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B_half, K * N * sizeof(half), cudaMemcpyHostToDevice);
    
    // Configure kernel launch
    const int warpsPerBlock = 4;
    const int threadsPerBlock = warpsPerBlock * 32;
    const int totalWarps = (M / WMMA_M) * (N / WMMA_N);
    const int blocksPerGrid = (totalWarps + warpsPerBlock - 1) / warpsPerBlock;
    
    std::cout << "\nKernel configuration:" << std::endl;
    std::cout << "Blocks: " << blocksPerGrid << ", Threads per block: " << threadsPerBlock << std::endl;
    
    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Launch GPU kernel
    std::cout << "\nRunning GPU WMMA kernel..." << std::endl;
    cudaEventRecord(start);
    wmma_gemm<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, M, N, K);
    cudaEventRecord(stop);
    
    // Wait for GPU to finish
    cudaEventSynchronize(stop);
    
    // Calculate GPU time
    float gpu_time;
    cudaEventElapsedTime(&gpu_time, start, stop);
    
    // Copy result back to host
    cudaMemcpy(h_C_gpu, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    
    // CPU verification
    std::cout << "Running CPU verification..." << std::endl;
    auto cpu_start = std::chrono::high_resolution_clock::now();
    cpu_gemm(h_A, h_B, h_C_cpu, M, N, K);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    auto cpu_time = std::chrono::duration_cast<std::chrono::milliseconds>(cpu_end - cpu_start).count();
    
    // Check results
    std::cout << "\nVerifying results..." << std::endl;
    bool results_match = check_results(h_C_gpu, h_C_cpu, M * N);
    
    if (results_match) {
        std::cout << "SUCCESS: GPU and CPU results match within tolerance!" << std::endl;
    } else {
        std::cout << "WARNING: Some results differ beyond tolerance (expected with half precision)" << std::endl;
    }
    
    // Performance metrics
    std::cout << "\nPerformance:" << std::endl;
    std::cout << "GPU Time (WMMA): " << std::fixed << std::setprecision(2) << gpu_time << " ms" << std::endl;
    std::cout << "CPU Time: " << cpu_time << " ms" << std::endl;
    std::cout << "Speedup: " << std::fixed << std::setprecision(1) 
              << (float)cpu_time / gpu_time << "x" << std::endl;
    
    // Calculate GFLOPS
    double flops = 2.0 * M * N * K;
    double gpu_gflops = (flops / (gpu_time / 1000.0)) / 1e9;
    double cpu_gflops = (flops / (cpu_time / 1000.0)) / 1e9;
    
    std::cout << "\nGPU Performance: " << std::fixed << std::setprecision(1) 
              << gpu_gflops << " GFLOPS" << std::endl;
    std::cout << "CPU Performance: " << std::fixed << std::setprecision(1) 
              << cpu_gflops << " GFLOPS" << std::endl;
    
    // Cleanup
    delete[] h_A;
    delete[] h_B;
    delete[] h_C_gpu;
    delete[] h_C_cpu;
    delete[] h_A_half;
    delete[] h_B_half;
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return 0;
}