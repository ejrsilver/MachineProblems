#include "cuda_runtime.h"
#include <stdio.h>
#include <iostream>
/*
Problem 1: Device Query
- # CUDA devices
- Device types
- Clock rate
- # streaming multiprocessors
- # cores
- warp size
- amount of global memory
- amount of constant memory
- amount of shared memory per block
- # registers available per block
- max # threads per block
- max size of each dimension of a block
- max size of each dimension of a grid
*/

int CoreCount (cudaDeviceProp prop) {
  int mul = prop.multiProcessorCount;
  switch (prop.major) {
    case 2:
      if (prop.minor == 1)
        return mul*48;
      else
        return mul*32;
      break;
    case 3:
      return mul*192;
      break;
    case 5:
      return mul*128;
      break;
    case 6:
      if (prop.minor == 1 || prop.minor == 2)
        return mul*128;
      else if (prop.minor == 0)
        return mul*64;
      break;
    case 7:
      if (prop.minor == 0 || prop.minor == 5)
        return mul*64;
      break;
  }
  return -1;
}

int main() {
  int n_devs;
  cudaGetDeviceCount(&n_devs);
  std::cout << "Number of CUDA devices: " << n_devs << std::endl;

  for(int x = 0; x < n_devs; x++) {
    cudaDeviceProp temp;
    cudaGetDeviceProperties(&temp, x);

    std::cout << "Name: " << temp.name << " :" << std::endl;
    std::cout << "Clock: " << temp.clockRate << " :" << std::endl;
    std::cout << "SMs: " << temp.multiProcessorCount << " :" << std::endl;
    std::cout << "Cores: " << CoreCount(temp) << " :" << std::endl;
    std::cout << "Warp Size: " << temp.warpSize << " :" << std::endl;
    std::cout << "Global Memory: " << temp.totalGlobalMem << " :" << std::endl;
    std::cout << "Constant Memory: " << temp.totalConstMem << std::endl;
    std::cout << "Shared Memory per Block: " << temp.sharedMemPerBlock << " :" << std::endl;
    std::cout << "Registers per Block:  " << temp.regsPerBlock << " :" << std::endl;
    std::cout << "Max Threads per Block: " << temp.maxThreadsPerBlock << " :" << std::endl;
    std::cout << "Max Dimension Size per Block: " << temp.maxThreadsDim << " :" << std::endl;
    std::cout << "Max Dimension Size per Grid: " << temp.maxGridSize << " :" << std::endl;
  }
}
