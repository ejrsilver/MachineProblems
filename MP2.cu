#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <time.h>
#include <random>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

/*
Problem 2: Matrix Addition

Steps:
1. Allocate input and output in host code
2. kernel with one output matrix element per thread (16x16 thread blocks)
3. kernel with one output matrix row (16 threads per block)
4. kernel with one output matrix column (16 threads per block)
5. Analyse pros and cons of above approaches

Create Random matrices A and B:
- 125x125
- 250x250
- 500x500
- 1000x1000
- 2000x2000

Analytics:
- kernel execution time
- execute multiple times and report averages/remove outliers
- compare CPU and GPU performance using graphs/tables

CUDA Events:
- start time
- stop time
- begin recording
- stop recording
- find elapsed time
- destroy events
*/

#define N 125
#define M_SIZE N * N
#define B_WIDTH 16
#define N_BLOCKS N / B_WIDTH

__global__ void matrixAdd(float* A, float* B, float* C, int n) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < n && y < n) {
		int i = y*n + x;
		C[i] = A[i] + B[i];
	}
	__syncthreads();
}

__global__ void matrixAddRow(float* A, float* B, float* C, int n) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;

	if (x < n) {
		for (int y = 0; y < n; y++) {
			int i = y*n + x;
			C[i] = A[i] + B[i];
		}
	}
	__syncthreads();
}

__global__ void matrixAddColumn(float* A, float* B, float* C, int n) {
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (y < n) {
		for (int x = 0; x < n; x++) {
			int i = y*n + x;
			C[i] = A[i] + B[i];
		}
	}
	__syncthreads();
}

void cudaAdd(const float* A, const float* B, float* C, int n, int mode) {
	int m_size = n*n;
	// allocate gpu pointers and check for errors
	float* dev_A;
	float* dev_B;
	float* dev_C;

	cudaError_t gpu_error = cudaMalloc((void**)&dev_A, m_size * sizeof(float));

	if (gpu_error != cudaSuccess) {
		std::cout << "Error allocating A" << std::endl;
	}
	gpu_error = cudaMalloc((void**)&dev_B, m_size * sizeof(float));
	if (gpu_error != cudaSuccess) {
		std::cout << "Error allocating B" << std::endl;
	}
	gpu_error = cudaMalloc((void**)&dev_C, m_size * sizeof(float));
	if (gpu_error != cudaSuccess) {
		std::cout << "Error allocating C" << std::endl;
	}
	// copy matrices to gpu
	gpu_error = cudaMemcpy(dev_A, A, m_size * sizeof(float), cudaMemcpyHostToDevice);

	if (gpu_error != cudaSuccess) {
		std::cout << "error allocating A video memory: " << cudaGetErrorString(gpu_error) << std::endl;
	}
	gpu_error = cudaMemcpy(dev_B, B, m_size * sizeof(float), cudaMemcpyHostToDevice);

	if (gpu_error != cudaSuccess) {
		std::cout << "error allocating B video memory: " << cudaGetErrorString(gpu_error) << std::endl;
	}
	cudaEvent_t start, stop;
	float gpu_time = 0;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	dim3 grid;
	dim3 block;

	int num_blocks = n / B_WIDTH;
	if (n%B_WIDTH) num_blocks++;

	// Add by element
  if (mode == 0) {
    block = dim3(B_WIDTH, B_WIDTH);
    grid = dim3(num_blocks, num_blocks);

    cudaEventRecord(start, 0);
    matrixAdd << <grid, block >> >(dev_A, dev_B, dev_C, n);
    gpu_error = cudaGetLastError();
    if (gpu_error != cudaSuccess) {
      std::cout << "Error during addition by element: " << cudaGetErrorString(gpu_error) << std::endl;
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu_time, start, stop);
    std::cout << "One thread per element addition (ms): " << gpu_time << std::endl;
  }
  // Add by row
  if (mode == 1) {
    cudaEventRecord(start, 0);
    matrixAddRow << <grid, block >> >(dev_A, dev_B, dev_C, n);

    gpu_error = cudaGetLastError();
    if (gpu_error != cudaSuccess) {
      std::cout << "Error during addition by row: " << cudaGetErrorString(gpu_error) << std::endl;
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu_time, start, stop);
    std::cout << "One thread per row addition (ms): " << gpu_time << std::endl;
  }
	// Add by column
  if (mode == 2) {
    cudaEventRecord(start, 0);
    matrixAddColumn << <grid, block >> >(dev_A, dev_B, dev_C, n);

    gpu_error = cudaGetLastError();
    if (gpu_error != cudaSuccess) {
      std::cout << "Error during addition by column: " << cudaGetErrorString(gpu_error) << std::endl;
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu_time, start, stop);
    std::cout << "One thread per column addition (ms): " << gpu_time << std::endl;
  }

	// Add by CPU
	float cpu_time = 0;
	cudaEventRecord(start, 0);

	for (int i = 0; i < m_size; i++) {
		float t = A[i] + B[i];
	}
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&cpu_time, start, stop);
	std::cout << "CPU add time (ms): " << cpu_time << std::endl;

	cudaDeviceSynchronize();

	gpu_error = cudaMemcpy(C, dev_C, m_size * sizeof(float), cudaMemcpyDeviceToHost);
	if (gpu_error != cudaSuccess) {
		std::cout << "error copying C back to host: " << cudaGetErrorString(gpu_error) << std::endl;
	}
	gpu_error = cudaFree(dev_A);
	if (gpu_error != cudaSuccess) {
		std::cout << "error freeing dev_A: " << cudaGetErrorString(gpu_error) << std::endl;
	}
	gpu_error = cudaFree(dev_B);
	if (gpu_error != cudaSuccess) {
		std::cout << "error freeing dev_B: " << cudaGetErrorString(gpu_error) << std::endl;
	}
	cudaError_t c_error = cudaFree(dev_C);
	if (c_error != cudaSuccess) {
		std::cout << "error freeing dev_C: " << cudaGetErrorString(c_error) << std::endl;
	}
}

int test(const float* A, const float* B, const float* C, int n) {
	for (int i = 0; i < n*n; i++) {
		if (A[i] + B[i] != C[i]) {
			std::cout << "ERROR AT [" << i / n << ", " << i%n << "]: Incorrect sum. Expected: " << A[i] << " + " << B[i] << " = " << (A[i] + B[i]) << ", Result: " << C[i] << std::endl;
			return 1;
		}
	}
	std::cout << "Test passed! Addition was successful." << std::endl;
	return 0;
}

int main() {
	srand(time(0));
	// init matrices and assign random values
	float* A = (float*)malloc(M_SIZE * sizeof(float));
	float* B = (float*)malloc(M_SIZE * sizeof(float));
	float* C = (float*)malloc(M_SIZE * sizeof(float));

	for (int i = 0; i < M_SIZE; i++) {
		A[i] = rand() % 1000 / 10.0;
		B[i] = rand() % 1000 / 10.0;
	}
	/*
	mode 0: Add by element
	mode 1: Add by row
	mode 2: Add by column
	*/
	cudaAdd(A, B, C, N, 0);

	test(A, B, C, N);

	free(A);
	free(B);
	free(C);

	return 0;
}