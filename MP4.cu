#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <time.h>
#include <random>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

/*
Problem 3: Matrix Multiplication
Sizes:
- 125x125
- 250x250
- 500x500
- 1000x1000
- 2000x2000
Steps:
- 2 square input matrices M and N, output matrix P
- Find host to device time
- Find device to host time
- Compare matrix multiplication for GPU and CPU (single block 1 thread for the block)
- if all time is accounted for, is multiplication on GPU always worth it?
Block Widths:
- 2
- 4
- 10
- 20
- 25
*/

__global__ void matrixMul(float* M, float* N, float* P, int n) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < n && y < n) {
		float temp = 0;
		for (int q = 0; q < n; q++) {
			temp += M[y*n + q] * N[q*n + x];
		}
		P[y*n + x] = temp;
	}
	__syncthreads();
}

void cudaMul(const float* M, const float* N, float* P, int n, int b_width) {
	int m_size = n*n;
	// allocate gpu pointers and check for errors
	float* dev_M;
	float* dev_N;
	float* dev_P;

	// Allocate GPU memory
	cudaError_t gpu_error;
	gpu_error = cudaMalloc((void**)&dev_M, m_size * sizeof(float));
	if (gpu_error != cudaSuccess) std::cout << "Error allocating A" << std::endl;
	gpu_error = cudaMalloc((void**)&dev_N, m_size * sizeof(float));
	if (gpu_error != cudaSuccess) std::cout << "Error allocating B" << std::endl;
	gpu_error = cudaMalloc((void**)&dev_P, m_size * sizeof(float));
	if (gpu_error != cudaSuccess)	std::cout << "Error allocating C" << std::endl;

	// Initialize timer functionality
	cudaEvent_t start, stop;
	float compute_time = 0;
	float h2d_time = 0;
	float d2h_time = 0;

	gpu_error = cudaEventCreate(&start);
	if (gpu_error != cudaSuccess) std::cout << "Error creating start event: " << cudaGetErrorString(gpu_error) << std::endl;
	gpu_error = cudaEventCreate(&stop);
	if (gpu_error != cudaSuccess) std::cout << "Error creating stop event: " << cudaGetErrorString(gpu_error) << std::endl;

	// copy matrices to gpu
	cudaEventRecord(start, 0);
	gpu_error = cudaMemcpy(dev_M, M, m_size * sizeof(float), cudaMemcpyHostToDevice);
	if (gpu_error != cudaSuccess) std::cout << "error allocating A video memory: " << cudaGetErrorString(gpu_error) << std::endl;
	gpu_error = cudaMemcpy(dev_N, N, m_size * sizeof(float), cudaMemcpyHostToDevice);
	if (gpu_error != cudaSuccess) std::cout << "error allocating B video memory: " << cudaGetErrorString(gpu_error) << std::endl;
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&h2d_time, start, stop);
	std::cout << "Copy Host to Device time (ms): " << h2d_time << std::endl;

	// Initialize grid and blocks
	int num_blocks = n / b_width;
	if (n%b_width) num_blocks++;
	dim3 block = dim3(b_width, b_width);
	dim3 grid = dim3(num_blocks, num_blocks);

	// Multiply with GPU
	cudaEventRecord(start, 0);
	matrixMul << <grid, block >> >(dev_M, dev_N, dev_P, n);
	gpu_error = cudaGetLastError();
	if (gpu_error != cudaSuccess) std::cout << "Error during addition by element: " << cudaGetErrorString(gpu_error) << std::endl;
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&compute_time, start, stop);
	std::cout << "GPU compute time (ms): " << compute_time << std::endl;

	cudaDeviceSynchronize();

	// Copy back to host
	cudaEventRecord(start, 0);
	gpu_error = cudaMemcpy(P, dev_P, m_size * sizeof(float), cudaMemcpyDeviceToHost);
	if (gpu_error != cudaSuccess) std::cout << "error copying P back to host: " << cudaGetErrorString(gpu_error) << std::endl;
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&d2h_time, start, stop);
	std::cout << "Copy Device to Host time (ms): " << d2h_time << std::endl;

	// Destroy start and stop events
	gpu_error = cudaEventDestroy(start);
	if (gpu_error != cudaSuccess) std::cout << "Error destroying start event: " << cudaGetErrorString(gpu_error) << std::endl;
	gpu_error = cudaEventDestroy(stop);
	if (gpu_error != cudaSuccess) std::cout << "Error destroying stop event: " << cudaGetErrorString(gpu_error) << std::endl;

	// Free GPU memory
	gpu_error = cudaFree(dev_M);
	if (gpu_error != cudaSuccess) std::cout << "error freeing dev_M: " << cudaGetErrorString(gpu_error) << std::endl;
	gpu_error = cudaFree(dev_N);
	if (gpu_error != cudaSuccess) std::cout << "error freeing dev_N: " << cudaGetErrorString(gpu_error) << std::endl;
	gpu_error = cudaFree(dev_P);
	if (gpu_error != cudaSuccess) std::cout << "error freeing dev_P: " << cudaGetErrorString(gpu_error) << std::endl;
}

int testMul(const float* M, const float* N, const float* P, int n) {
	for (int x = 0; x < n; x++) {
		for (int y = 0; y < n; y++) {
			float temp = 0;
			for (int q = 0; q < n; q++) {
				temp += M[x*n + q] * N[q*n + y];
			}
			if (P[x*n + y] != temp) {
				std::cout << "ERROR AT [" << x << ", " << y << "]: Incorrect sum. Expected: " << temp << ", Result: " << P[x*n + y] << std::endl;
				return 1;
			}
		}
	}
	return 0;
}

int cpuMul(const float* M, const float* N, float* P, int n) {
	cudaEvent_t start, stop;
	float cpu_time = 0;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// Multiply with CPU
	cudaEventRecord(start, 0);
	for (int x = 0; x < n; x++) {
		for (int y = 0; y < n; y++) {
			float temp = 0;
			for (int q = 0; q < n; q++) {
				temp += M[y*n + q] * N[q*n + x];
			}
			P[x*n + y] = temp;
		}
	}
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&cpu_time, start, stop);
	std::cout << "CPU compute time (ms): " << cpu_time << std::endl;

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	return 0;
}

void MatrixNMul(int n) {
	float* M;
	float* N;
	float* P;

	// Initialize 125 x 125 matrix
	int m_size = n*n;
	int testout = 0;
	// init matrices and assign random values
	M = (float*)malloc(m_size * sizeof(float));
	N = (float*)malloc(m_size * sizeof(float));
	P = (float*)malloc(m_size * sizeof(float));

	for (int i = 0; i < m_size; i++) {
		M[i] = rand() % 1000 / 10.0;
		N[i] = rand() % 1000 / 10.0;
	}
	std::cout << "Testing for matrix size: " << n << " x " << n << std::endl;

	// Multiply
	std::cout << "Block size: " << 1 << std::endl;
	cudaMul(M, N, P, n, 1);
	testout += testMul(M, N, P, n);
	std::cout << "Block size: " << 2 << std::endl;
	cudaMul(M, N, P, n, 2);	
	testout += testMul(M, N, P, n);
	std::cout << "Block size: " << 4 << std::endl;
	cudaMul(M, N, P, n, 4);
	testout += testMul(M, N, P, n);
	std::cout << "Block size: " << 10 << std::endl;
	cudaMul(M, N, P, n, 10);
	testout += testMul(M, N, P, n);
	std::cout << "Block size: " << 20 << std::endl;
	cudaMul(M, N, P, n, 20);
	testout += testMul(M, N, P, n);
	std::cout << "Block size: " << 25 << std::endl;
	cudaMul(M, N, P, n, 25);
	testout += testMul(M, N, P, n);
	cpuMul(M, N, P, n);
	testout += testMul(M, N, P, n);
	if (testout == 0) std::cout << "Test passed!" << std::endl;
	else std::cout << "Test failed!" << std::endl;

	free(M);
	free(N);
	free(P);
}

int main() {
	// init pseudorandom generator so that A and B will not be identical every time
	srand(time(0));

	// 125x125
	MatrixNMul(125);
	// 250x250
	MatrixNMul(250);
	// 500x500
	MatrixNMul(500);
	//1000x1000
	MatrixNMul(1000);
	//2000x2000
	MatrixNMul(2000);

	return 0;
}