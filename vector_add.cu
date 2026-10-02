#include <iostream>
#include <cmath>
#include <chrono>
#include <cuda_runtime.h>

// ------------------------------------------------------------
// CPU version
// ------------------------------------------------------------
void vector_add_cpu(const float* A, const float* B, float* C, int N)
{
    for (int i = 0; i < N; ++i) {
        C[i] = A[i] + B[i];
    }
}

// ------------------------------------------------------------
// GPU kernel
// One CUDA thread handles one element.
// ------------------------------------------------------------
__global__ void vector_add_gpu(const float* A, const float* B, float* C, int N)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

// ------------------------------------------------------------
// Error-checking helper
// ------------------------------------------------------------
#define CUDA_CHECK(call)                                                   \
do {                                                                       \
    cudaError_t err = (call);                                              \
    if (err != cudaSuccess) {                                              \
        std::cerr << "CUDA error: " << cudaGetErrorString(err)             \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl;  \
        return 1;                                                          \
    }                                                                      \
} while (0)

int main()
{
    // Use a larger N so that timing is meaningful.
    int N = 10'000'000;

    size_t bytes = N * sizeof(float);

    // ------------------------------------------------------------
    // Host (CPU) memory
    // ------------------------------------------------------------
    float* A = new float[N];
    float* B = new float[N];
    float* C_cpu = new float[N];
    float* C_gpu = new float[N];

    for (int i = 0; i < N; ++i) {
        A[i] = (float)i + 1.0f;
        B[i] = 2.0f;
    }

    // ------------------------------------------------------------
    // CPU operation + timing
    // ------------------------------------------------------------
    auto cpu_start = std::chrono::high_resolution_clock::now();

    vector_add_cpu(A, B, C_cpu, N);

    auto cpu_end = std::chrono::high_resolution_clock::now();

    double cpu_ms =
        std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();

    // ------------------------------------------------------------
    // Device (GPU) memory
    // ------------------------------------------------------------
    float *d_a, *d_b, *d_c;

    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));

    CUDA_CHECK(cudaMemcpy(d_a, A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, B, bytes, cudaMemcpyHostToDevice));

    // ------------------------------------------------------------
    // CUDA launch configuration
    // ------------------------------------------------------------
    int blocksize = 256;
    int gridsize = (N + blocksize - 1) / blocksize;

    // ------------------------------------------------------------
    // GPU timing
    //
    // CUDA events measure time on the GPU much more appropriately
    // than std::chrono around the kernel launch.
    // ------------------------------------------------------------
    cudaEvent_t start, stop;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));

    vector_add_gpu<<<gridsize, blocksize>>>(d_a, d_b, d_c, N);

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    CUDA_CHECK(cudaGetLastError());

    float gpu_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&gpu_ms, start, stop));

    // Copy result back to CPU.
    CUDA_CHECK(cudaMemcpy(C_gpu, d_c, bytes, cudaMemcpyDeviceToHost));

    // ------------------------------------------------------------
    // Verify GPU result against CPU result
    // ------------------------------------------------------------
    bool correct = true;

    for (int i = 0; i < N; ++i) {
        if (C_cpu[i] != C_gpu[i]) {
            std::cout << "Mismatch at index " << i
                      << ": CPU=" << C_cpu[i]
                      << ", GPU=" << C_gpu[i] << "\n";
            correct = false;
            break;
        }
    }

    // ------------------------------------------------------------
    // Print information
    // ------------------------------------------------------------
    std::cout << "\n========== CUDA Vector Addition ==========\n";

    std::cout << "N                  : " << N << "\n";
    std::cout << "Block size         : " << blocksize << " threads\n";
    std::cout << "Grid size          : " << gridsize << " blocks\n";
    std::cout << "Threads launched   : "
              << (long long)gridsize * blocksize << "\n";

    std::cout << "Useful threads     : " << N << "\n";
    std::cout << "Threads outside N  : "
              << (long long)gridsize * blocksize - N << "\n";

    std::cout << "\nCPU time           : " << cpu_ms << " ms\n";
    std::cout << "GPU kernel time    : " << gpu_ms << " ms\n";

    if (gpu_ms > 0.0f) {
        std::cout << "CPU/GPU time ratio : "
                  << cpu_ms / gpu_ms << "x\n";
    }

    std::cout << "Results match      : "
              << (correct ? "YES" : "NO") << "\n";

    std::cout << "\nFirst 10 results:\n";
    for (int i = 0; i < 10; ++i) {
        std::cout << "C[" << i << "] = " << C_gpu[i] << "\n";
    }

    // ------------------------------------------------------------
    // Cleanup
    // ------------------------------------------------------------
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));

    delete[] A;
    delete[] B;
    delete[] C_cpu;
    delete[] C_gpu;

    return 0;
}
