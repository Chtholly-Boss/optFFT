#include <torch/extension.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <device_launch_parameters.h>
#include <corecrt_math_defines.h>

#define N 8

__global__
void
fft_naive(cuFloatComplex* x, int n, int steps){
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	for (int stage = 0; stage < steps; ++stage){
		int numOfElem = 1 << (stage + 1);
		int xStart = idx * numOfElem;
		if (xStart < n){
			for(int k = 0; k < numOfElem/2; ++k){
				// Butterfly Operation
				float angle = -2.0f * M_PI * k / numOfElem;
				cuFloatComplex rotation = make_cuFloatComplex(cos(angle), sin(angle));

				cuFloatComplex even = x[xStart+k];
				cuFloatComplex odd = x[xStart + numOfElem/2 + k];
				cuFloatComplex twiddle = cuCmulf(rotation,odd);
				x[xStart+k] = cuCaddf(even,twiddle);
				x[xStart + numOfElem/2 + k] = cuCsubf(even, twiddle);  
			}
		}
		__syncthreads();
	}
}

/**
 * @brief reverse the given number's bit representation
 * 
 * @param num 	the num to be processed
 * @param bits 	the width to be processed 
 * @return int 	the reversed num
 */
int
reverse_bit(int num, int bits){
	int res = 0;
	for (int i = 0; i < bits; ++i){
		res = (res << 1) | (num & 1);
		num >>= 1;
	}
	return res;
}

/**
 * @brief 	bit-reverse a given array
 * 
 * @param arr 
 * @param n	length of the array
 * @return int status, 0 for success
 */
int 
bit_reverse(cuFloatComplex* arr, int n){
	// get the bits needed to represent the length
	int bits = 0;
	while ((1 << bits) < n){
		bits++;
	}
	// swap the reverse-bits pair
	for (int i = 0; i < n; ++i){
		int j = reverse_bit(i,bits);
		if (i < j){
			cuFloatComplex temp = arr[i];
			arr[i] = arr[j];
			arr[j] = temp;
		}
	}
	return 0;
}

/**
 * @brief 
 * 
 * @param res 
 * @param n 
 * @return int 
 */
int print_res(cuFloatComplex* res, int n){
	printf("Index\tValue\n");
	for(int i = 0; i < n; ++i){
		printf("[%d]:\t(%.2f, %.2f)\n", i, res[i].x, res[i].y);
	}
	printf("\n");
	return 0;
}

int fft(cuFloatComplex* x_h, int n){
	cuFloatComplex* x_d;
	// Allocate and Copy
	int sz = N * sizeof cuFloatComplex;
	cudaMalloc((void**)&x_d, sz);
	cudaMemcpy(x_d, x_h,sz,cudaMemcpyHostToDevice);	
	// Launch the Kernel
	fft_naive<<<1,512>>>(x_d, n, log2(n));
	// Post process
	cudaMemcpy(x_h,x_d,sz,cudaMemcpyDeviceToHost);
	cudaFree(x_d);
	return 0;
}

/**
 * @brief 
 * 
 * @param argc 
 * @param argv 
 * @return int 
 */
int main(int argc, char* argv[]) {
	
	cuFloatComplex *x_h;
	x_h = (cuFloatComplex*) malloc(N * sizeof cuFloatComplex);
	// initialize 
	for (int i = 0; i < N; ++i){
		x_h[i] = make_cuFloatComplex((float)i,0.0);
	}
	print_res(x_h, N);
	// Do Bit Reverse in the host
	bit_reverse(x_h,N);
	fft(x_h, N);
	print_res(x_h,N);
	// free host memory
	free(x_h);
	return 0;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fft_naive", &fft_naive, "FFT naive implementation(CUDA)");
}