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

#define M_DIM 125
#define M_SIZE M_DIM * M_DIM
#define B_WIDTH 16
#define N_BLOCKS M_DIM * B_WIDTH

__global__ void matrixAdd(float A[N][N], float B[N][N], float C[N][N]) {
  int i = threadIdx.x;
  int j = threadIdx.y;
  C[i][j] = A[i][j] + B[i][j];
}

__global__ void matrixAddRow(float A[N], float B[N], float C[N]) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
}

int main() {
  srand(time(0));
  // init matrices and assign random values
  float** A = (float**) malloc(M_SIZE * sizeof(float*)); 
  float** B = (float**) malloc(M_SIZE * sizeof(float*)); 
  float** C = (float**) malloc(M_SIZE * sizeof(float*));

  for(int x = 0; x < M_DIM; x++) {
    for(int y = 0; y < M_DIM; y++) {
      A[i][j] = rand() % 100;
      B[i][j] = rand() % 100;
    }
  }

  // allocate gpu pointers and check for errors
  float** dev_A;
  float** dev_B;
  float** dev_C;
  
  cudaError_t gpu_error = cudaMalloc((void**) &dev_a, M_SIZE * sizeof(float*));
  if (gpu_error != cudaSuccess) {
    std::cout << "Error allocating A" << std::endl;
  }
  gpu_error = cudaMalloc((void**) &dev_b, M_SIZE * sizeof(float*));
  if (gpu_error != cudaSuccess) {
    std::cout << "Error allocating B" << std::endl;
  }
  gpu_error = cudaMalloc((void**) &dev_c, M_SIZE * sizeof(float*));
  if (gpu_error != cudaSuccess) {
    std::cout << "Error allocating C" << std::endl;
  }
  // copy matrices to gpu
  cudaMemcpy(dev_A, A, M_SIZE * sizeof(float*), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_B, B, M_SIZE * sizeof(float*), cudaMemcpyHostToDevice);

  cudaEvent_t start, stop;
  float gpu_time = 0;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  dim3 grid;
  dim3 block;

  // Add by element
  if(M_DIM % B_WIDTH) {
    grid = dim3(N_BLOCKS+1, N_BLOCKS+1);
  }
  else {
    grid = dim3(N_BLOCKS, N_BLOCKS);
  }
  block = dim3(B_WIDTH, B_WIDTH);

  cudaEventRecord(start, 0);
  matrixAdd<<<grid, block>>>(A, B, C);

  gpu_error = cudaGetLastError();
  if (gpu_error != cudaSuccess) {
    std::cout << "Error during addition by element: " << cudaGetErrorString(gpu_error) << std::endl;
  }
  
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&gpu_time, start, stop);

  std::cout << "One thread per element addition (ms): " << gpu_time << std::endl;

  // Add by row
  block = dim3(B_WIDTH/4, B_WIDTH/4);

  cudaEventRecord(start, 0);
  matrixAddRow<<<grid, block>>>(A, B, C);

  gpu_error = cudaGetLastError();
  if (gpu_error != cudaSuccess) {
    std::cout << "Error during addition by row: " << cudaGetErrorString(gpu_error) << std::endl;
  }
  
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&gpu_time, start, stop);
  
  std::cout << "One thread per row addition (ms): " << gpu_time << std::endl;

  // Add by column
  block = dim3(B_WIDTH/4, B_WIDTH/4);

  cudaEventRecord(start, 0);
  matrixAddColumn<<<grid, block>>>(A, B, C);

  gpu_error = cudaGetLastError();
  if (gpu_error != cudaSuccess) {
    std::cout << "Error during addition by column: " << cudaGetErrorString(gpu_error) << std::endl;
  }
  
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&gpu_time, start, stop);
  
  std::cout << "One thread per column addition (ms): " << gpu_time << std::endl;

  // Add by CPU
  float cpu_time = 0;
  cudaEventRecord(start, 0);

  for(int x = 0; x < M_DIM; x++) {
    for(int y = 0; y < M_DIM; y++) {
      float t = A[i][j] + B[i][j];
    }
  }
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&cpu_time, start, stop);
  std::cout << "CPU add time (ms): " << cpu_time << std::endl;

  cudaDeviceSynchronize();

  gpu_error = cudaMemcpy(C, dev_C, M_SIZE * sizeof(float*), cudaMemcpyDeviceToHost);
  if (gpu_error != cudaSuccess) {
    std::cout << "error copying C back to host" << std::endl;
  }

	gpu_error = cudaFree(dev_A);
	if (gpu_error != cudaSuccess) {
    std::cout << "error freeing dev_A" << std::endl;
  }
	gpu_error = cudaFree(dev_B);
	if (gpu_error != cudaSuccess) {
    std::cout << "error freeing dev_B" << std::endl;
  }
	gpu_error = cudaFree(dev_C);
	if (gpu_error != cudaSuccess) {
    std::cout << "error freeing dev_C" << std::endl;
  }

  free(A);
  free(B);
  free(C);

  return 0;
}