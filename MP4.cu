#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <time.h>
#include <random>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define TILE_WIDTH 10

/*
Problem 4: Tiled Matrix Multiplication
Sizes:
- 125x125
- 250x250
- 500x500
- 1000x1000
- 2000x2000
Bonus: (tile size 8x15)
- 350x400 400x500
- 1900x1600 1600x1300
Tile Widths:
- 2
- 5
- 10
- 20
- 25
*/

__global__ void matrixMul(float* M, float* N, float* P, int Mr, int Mc, int Nr, int Nc, int Cr, int Cc) {
	__shared__ float M_s[TILE_WIDTH][TILE_WIDTH];
	__shared__ float N_s[TILE_WIDTH][TILE_WIDTH];

	int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
	int col = blockIdx.x * TILE_WIDTH + threadIdx.x;
	float temp = 0;

	for (int i = 0; i < (TILE_WIDTH + Mc - 1) / TILE_WIDTH; i++) {
		if (i*TILE_WIDTH + threadIdx.x < Mc && row < Mr) {
			M_s[threadIdx.y][threadIdx.x] = M[row*Mc + i*TILE_WIDTH + threadIdx.x];
		}
		else {
			M_s[threadIdx.y][threadIdx.x] = 0.0;
		}

		if (i*TILE_WIDTH + threadIdx.y < Nr && col < Nc) {
			N_s[threadIdx.y][threadIdx.x] = N[(i*TILE_WIDTH + threadIdx.y)*Nc + col];
		}
		else {
			N_s[threadIdx.y][threadIdx.x] = 0.0;
		}
		__syncthreads();

		for (int q = 0; q < TILE_WIDTH; ++q) {
			temp += M_s[threadIdx.y][q] * N_s[q][threadIdx.x];
		}
		__syncthreads();
	}
	if (row < Cr && col < Cc) {
		P[(blockIdx.y*blockDim.y + threadIdx.y)*Cc + blockIdx.x*blockDim.x + threadIdx.x] = temp;
	}
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
				temp += M[x*n + q] * N[q*n + y];
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

int testMul(const float* M, const float* N, const float* P, int n) {
	for (int x = 0; x < n; x++) {
		for (int y = 0; y < n; y++) {
			float temp = 0;
			for (int q = 0; q < n; q++) {
				temp += M[x*n + q] * N[q*n + y];
			}
			if (P[x*n + y] != temp) {
				std::cout << "ERROR AT [" << x << ", " << y << "]: Incorrect product. Expected: " << temp << ", Result: " << P[x*n + y] << std::endl;
				return 1;
			}
		}
	}
	return 0;
}

void cudaMul(const float* M, const float* N, float* P, int n) {
	int m_size = n*n;
	// allocate gpu pointers and check for errors
	float* dev_M;
	float* dev_N;
	float* dev_P;

	// Allocate GPU memory
	cudaError_t gpu_error;
	gpu_error = cudaMalloc((void**)&dev_M, m_size * sizeof(float));
	if (gpu_error != cudaSuccess) std::cout << "Error allocating M" << std::endl;
	gpu_error = cudaMalloc((void**)&dev_N, m_size * sizeof(float));
	if (gpu_error != cudaSuccess) std::cout << "Error allocating N" << std::endl;
	gpu_error = cudaMalloc((void**)&dev_P, m_size * sizeof(float));
	if (gpu_error != cudaSuccess)	std::cout << "Error allocating P" << std::endl;

	// Initialize timer functionality
	cudaEvent_t start, stop;
	float compute_time = 0;

	gpu_error = cudaEventCreate(&start);
	if (gpu_error != cudaSuccess) std::cout << "Error creating start event: " << cudaGetErrorString(gpu_error) << std::endl;
	gpu_error = cudaEventCreate(&stop);
	if (gpu_error != cudaSuccess) std::cout << "Error creating stop event: " << cudaGetErrorString(gpu_error) << std::endl;

	// copy matrices to gpu
	gpu_error = cudaMemcpy(dev_M, M, m_size * sizeof(float), cudaMemcpyHostToDevice);
	if (gpu_error != cudaSuccess) std::cout << "error allocating M video memory: " << cudaGetErrorString(gpu_error) << std::endl;
	gpu_error = cudaMemcpy(dev_N, N, m_size * sizeof(float), cudaMemcpyHostToDevice);
	if (gpu_error != cudaSuccess) std::cout << "error allocating N video memory: " << cudaGetErrorString(gpu_error) << std::endl;

	// Initialize grid and blocks
	int num_blocks = n / TILE_WIDTH;
	if (n%TILE_WIDTH) num_blocks++;
	dim3 block = dim3(TILE_WIDTH, TILE_WIDTH);
	dim3 grid = dim3(num_blocks, num_blocks);

	// Multiply with GPU
	cudaEventRecord(start, 0);
	matrixMul << <grid, block >> >(dev_M, dev_N, dev_P, n, n, n, n, n, n);
	gpu_error = cudaGetLastError();
	if (gpu_error != cudaSuccess) std::cout << "Error during multiplication: " << cudaGetErrorString(gpu_error) << std::endl;
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&compute_time, start, stop);
	std::cout << "GPU compute time (ms): " << compute_time << std::endl;

	cudaDeviceSynchronize();

	// Copy back to host
	gpu_error = cudaMemcpy(P, dev_P, m_size * sizeof(float), cudaMemcpyDeviceToHost);
	if (gpu_error != cudaSuccess) std::cout << "error copying P back to host: " << cudaGetErrorString(gpu_error) << std::endl;

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
	std::cout << "Tile size: " << TILE_WIDTH << std::endl;
	cudaMul(M, N, P, n);
	testout += testMul(M, N, P, n);
	cpuMul(M, N, P, n);
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
