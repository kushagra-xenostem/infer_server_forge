#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>

__global__ void vector_add(const float* a, const float* b, float* c, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        c[i] = a[i] + b[i];
}

int main()
{
    int n = 1024 * 1024;
    size_t bytes = n * sizeof(float);

    std::vector<float> h_a(n, 1.5f);
    std::vector<float> h_b(n, 2.5f);
    std::vector<float> h_c(n, 0.0f);

    float *d_a, *d_b, *d_c;

    hipError_t err;

    err = hipMalloc(&d_a, bytes);
    if (err != hipSuccess) {
        std::cerr << "hipMalloc d_a failed: "
                  << hipGetErrorString(err) << "\n";
        return 1;
    }

    err = hipMalloc(&d_b, bytes);
    if (err != hipSuccess) {
        std::cerr << "hipMalloc d_b failed: "
                  << hipGetErrorString(err) << "\n";
        return 1;
    }

    err = hipMalloc(&d_c, bytes);
    if (err != hipSuccess) {
        std::cerr << "hipMalloc d_c failed: "
                  << hipGetErrorString(err) << "\n";
        return 1;
    }

    hipMemcpy(d_a, h_a.data(), bytes, hipMemcpyHostToDevice);
    hipMemcpy(d_b, h_b.data(), bytes, hipMemcpyHostToDevice);

    vector_add<<<(n + 255) / 256, 256>>>(d_a, d_b, d_c, n);

    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Kernel launch failed: "
                  << hipGetErrorString(err) << "\n";
        return 1;
    }

    err = hipDeviceSynchronize();
    if (err != hipSuccess) {
        std::cerr << "Kernel execution failed: "
                  << hipGetErrorString(err) << "\n";
        return 1;
    }

    hipMemcpy(h_c.data(), d_c, bytes, hipMemcpyDeviceToHost);

    bool correct = true;

    for (int i = 0; i < n; ++i) {
        if (std::fabs(h_c[i] - 4.0f) > 1e-5f) {
            correct = false;
            std::cerr << "Wrong result at " << i
                      << ": " << h_c[i] << "\n";
            break;
        }
    }

    std::cout << "GPU: ";
    hipDeviceProp_t prop;
    hipGetDeviceProperties(&prop, 0);
    std::cout << prop.name << "\n";

    std::cout << "GFX: " << prop.gcnArchName << "\n";

    std::cout << "Vector addition: "
              << (correct ? "PASS" : "FAIL") << "\n";

    hipFree(d_a);
    hipFree(d_b);
    hipFree(d_c);

    return correct ? 0 : 1;
}
