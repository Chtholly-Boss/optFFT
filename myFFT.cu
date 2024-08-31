#include <torch/extension.h>
#include <math.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>

typedef struct {
    float real;
    float imag;
} Complex;

static __device__ __host__ inline Complex ComplexAdd(Complex, Complex);
static __device__ __host__ inline Complex ComplexSub(Complex a, Complex b);
static __device__ __host__ inline Complex ComplexMul(Complex, Complex);
static __device__ __host__ inline Complex make_Complex(float a, float b);

////////////////////////////////////////////////////////////////////////////////
// FFT Naive Implemntation
////////////////////////////////////////////////////////////////////////////////
__global__
void
fft_naive_cu(Complex* x, int n, int steps){
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	for (int stage = 0; stage < steps; ++stage){
		int numOfElem = 1 << (stage + 1);
		int xStart = idx * numOfElem;
		if (xStart < n){
			for(int k = 0; k < numOfElem/2; ++k){
				// Butterfly Operation
				float angle = -2.0f * M_PI * k / numOfElem;
				Complex rotation = make_Complex(cos(angle), sin(angle));

				Complex even = x[xStart+k];
				Complex odd = x[xStart + numOfElem/2 + k];
				Complex twiddle = ComplexMul(rotation,odd);
				x[xStart+k] = ComplexAdd(even,twiddle);
				x[xStart + numOfElem/2 + k] = ComplexSub(even, twiddle);  
			}
		}
		__syncthreads();
	}
}

torch::Tensor fft_naive(torch::Tensor x){
	// Get length	
	int n = x.size(0);
	// launch kernel
	int threads = 1024;
	int blocks = (n + threads - 1) / threads;
	Complex* x_ptr = reinterpret_cast<Complex*>(x.data_ptr());
	fft_naive_cu<<<blocks, threads>>>(x_ptr, n, log2(n));
	cudaDeviceSynchronize();
	return x;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("naive", &fft_naive, "My FFT naive implementation(CUDA)");
}

////////////////////////////////////////////////////////////////////////////////
// Complex Functions
////////////////////////////////////////////////////////////////////////////////
static __device__ __host__ inline Complex make_Complex(float a, float b) {
  Complex c;
  c.real = a;
  c.imag = b;
  return c;
}
static __device__ __host__ inline Complex ComplexAdd(Complex a, Complex b) {
  Complex c;
  c.real = a.real + b.real;
  c.imag = a.imag + b.imag;
  return c;
}

static __device__ __host__ inline Complex ComplexSub(Complex a, Complex b) {
  Complex c;
  c.real = a.real - b.real;
  c.imag = a.imag - b.imag;
  return c;
}

static __device__ __host__ inline Complex ComplexMul(Complex a, Complex b) {
  Complex c;
  c.real = a.real * b.real - a.imag * b.imag;
  c.imag = a.real * b.imag + a.imag * b.real;
  return c;
}
