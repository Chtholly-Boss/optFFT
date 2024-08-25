#include <stdio.h>
#include <cuda_runtime.h>
#include <cuComplex.h>

__device__ int bit_reverse(int n, int bits) {
    int result = 0;
    for (int i = 0; i < bits; i++) {
        result <<= 1;            // Shift result left by 1
        result |= (n & 1);      // Get the least significant bit of n and add it to result
        n >>= 1;                // Shift n right by 1
    }
    return result;
}

__global__ void fft_iterative(cuFloatComplex* x, int N, int logN) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Bit-reverse the indices
    if (idx < N) {
        int bit_reversed_index = bit_reverse(idx, logN);
        if (bit_reversed_index > idx) {
            cuFloatComplex temp = x[idx];
            x[idx] = x[bit_reversed_index];
            x[bit_reversed_index] = temp;
        }
    }
    __syncthreads(); // Synchronize after bit-reversing

    // FFT computation
    for (int s = 1; s <= logN; s++) {
        int m = 1 << s;
        cuFloatComplex wm = make_cuFloatComplex(cos(-2.0f * M_PI / m), sin(-2.0f * M_PI / m)); // Rotation factor

        for (int k = idx; k < N; k += m) {
            cuFloatComplex w = make_cuFloatComplex(1.0f, 0.0f); // Initialize w
            for (int j = 0; j < m / 2; j++) {
                cuFloatComplex t = cuCmulf(w, x[k + j + m / 2]);
                cuFloatComplex u = x[k + j];
                x[k + j] = cuCaddf(u, t);
                x[k + j + m / 2] = cuCsubf(u, t);
                w = cuCmulf(w, wm); // Update w
            }
        }
        __syncthreads(); // Synchronize after each stage
    }
}

void fft(cuFloatComplex* h_x, int N) {
    cuFloatComplex* d_x;
    int logN = log2(N);

    // Allocate device memory
    cudaMalloc((void**)&d_x, N * sizeof(cuFloatComplex));
    cudaMemcpy(d_x, h_x, N * sizeof(cuFloatComplex), cudaMemcpyHostToDevice);

    // Launch kernel for bit-reversal and FFT
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    fft_iterative<<<blocksPerGrid, threadsPerBlock>>>(d_x, N, logN);
    cudaDeviceSynchronize();

    // Copy results back to host
    cudaMemcpy(h_x, d_x, N * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_x);
}

int main() {
    // Example usage
    int N = 8; // Size of the input
    cuFloatComplex h_x[] = {
        make_cuFloatComplex(1.0f, 0.0f),
        make_cuFloatComplex(0.0f, 0.0f),
        make_cuFloatComplex(0.0f, 0.0f),
        make_cuFloatComplex(0.0f, 0.0f),
        make_cuFloatComplex(0.0f, 0.0f),
        make_cuFloatComplex(0.0f, 0.0f),
        make_cuFloatComplex(0.0f, 0.0f),
        make_cuFloatComplex(0.0f, 0.0f)
    };

    fft(h_x, N);

    // Print results
    for (int i = 0; i < N; i++) {
        printf("h_x[%d] = (%f, %f)\n", i, cuCrealf(h_x[i]), cuCimagf(h_x[i]));
    }

    return 0;
}