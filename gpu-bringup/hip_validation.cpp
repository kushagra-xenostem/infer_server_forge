#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

#define HIP_CHECK(call)                                                     \
    do {                                                                     \
        hipError_t err__ = (call);                                           \
        if (err__ != hipSuccess) {                                           \
            std::cerr << "HIP ERROR: " << #call << "\n"                      \
                      << "  " << hipGetErrorString(err__) << "\n";           \
            return 1;                                                        \
        }                                                                    \
    } while (0)

__global__ void saxpy_kernel(
    const float* x,
    const float* y,
    float* out,
    float a,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n)
        out[i] = a * x[i] + y[i];
}

__global__ void scale_kernel(float* data, float scale, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n)
        data[i] *= scale;
}

int main()
{
    int device_count = 0;
    HIP_CHECK(hipGetDeviceCount(&device_count));

    if (device_count <= 0) {
        std::cerr << "No HIP devices found\n";
        return 1;
    }

    int device = 0;
    HIP_CHECK(hipSetDevice(device));

    hipDeviceProp_t prop{};
    HIP_CHECK(hipGetDeviceProperties(&prop, device));

    std::cout << "Device: " << prop.name << "\n";
    std::cout << "GFX: " << prop.gcnArchName << "\n";
    std::cout << "Compute Units: " << prop.multiProcessorCount << "\n";
    std::cout << "Global Memory: "
              << static_cast<double>(prop.totalGlobalMem) / (1024.0 * 1024.0 * 1024.0)
              << " GiB\n";

    const int n = 16 * 1024 * 1024;
    const size_t bytes = static_cast<size_t>(n) * sizeof(float);

    std::vector<float> h_x(n, 2.0f);
    std::vector<float> h_y(n, 3.0f);
    std::vector<float> h_out(n, 0.0f);

    float* d_x = nullptr;
    float* d_y = nullptr;
    float* d_out = nullptr;

    HIP_CHECK(hipMalloc(&d_x, bytes));
    HIP_CHECK(hipMalloc(&d_y, bytes));
    HIP_CHECK(hipMalloc(&d_out, bytes));

    HIP_CHECK(hipMemcpy(
        d_x, h_x.data(), bytes, hipMemcpyHostToDevice));

    HIP_CHECK(hipMemcpy(
        d_y, h_y.data(), bytes, hipMemcpyHostToDevice));

    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;

    // Warm-up.
    hipLaunchKernelGGL(
        saxpy_kernel,
        dim3(grid_size),
        dim3(block_size),
        0,
        0,
        d_x,
        d_y,
        d_out,
        2.0f,
        n
    );

    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    // Time repeated GPU execution.
    hipEvent_t start{}, stop{};

    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));

    HIP_CHECK(hipEventRecord(start));

    constexpr int iterations = 100;

    for (int i = 0; i < iterations; ++i) {
        hipLaunchKernelGGL(
            saxpy_kernel,
            dim3(grid_size),
            dim3(block_size),
            0,
            0,
            d_x,
            d_y,
            d_out,
            2.0f,
            n
        );

        HIP_CHECK(hipGetLastError());
    }

    HIP_CHECK(hipEventRecord(stop));
    HIP_CHECK(hipEventSynchronize(stop));

    float elapsed_ms = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&elapsed_ms, start, stop));

    // Second kernel to verify chained execution.
    hipLaunchKernelGGL(
        scale_kernel,
        dim3(grid_size),
        dim3(block_size),
        0,
        0,
        d_out,
        0.5f,
        n
    );

    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(hipMemcpy(
        h_out.data(), d_out, bytes, hipMemcpyDeviceToHost));

    // Expected:
    // SAXPY = 2 * 2 + 3 = 7
    // scale by 0.5 = 3.5
    constexpr float expected = 3.5f;

    float max_error = 0.0f;

    for (int i = 0; i < n; ++i) {
        max_error = std::max(
            max_error,
            std::fabs(h_out[i] - expected)
        );
    }

    const double total_gb =
        static_cast<double>(bytes) * 3.0 * iterations /
        (1024.0 * 1024.0 * 1024.0);

    const double seconds =
        static_cast<double>(elapsed_ms) / 1000.0;

    const double effective_gbps =
        total_gb / seconds;

    std::cout << "\n===== RESULTS =====\n";
    std::cout << "Iterations: " << iterations << "\n";
    std::cout << "GPU time: " << elapsed_ms << " ms\n";
    std::cout << "Effective memory throughput: "
              << effective_gbps << " GB/s\n";
    std::cout << "Max numerical error: " << max_error << "\n";

    bool pass = max_error < 1e-5f;

    std::cout << "HIP validation: "
              << (pass ? "PASS" : "FAIL")
              << "\n";

    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));

    HIP_CHECK(hipFree(d_x));
    HIP_CHECK(hipFree(d_y));
    HIP_CHECK(hipFree(d_out));

    return pass ? 0 : 1;
}
