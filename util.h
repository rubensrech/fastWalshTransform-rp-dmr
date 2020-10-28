#ifndef UTIL_H
#define UTIL_H

#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>

#include <vector>

#define BLOCK_SIZE 32

#define CHECK_CUDA_ERROR(ans) { checkCudaError((ans), __FILE__, __LINE__); }
inline void checkCudaError(cudaError_t code, const char *file, int line, bool abort=false) {
    cudaDeviceSynchronize();
    if (code != cudaSuccess) {
        printf("CUDA Error: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

float find_max(float *array, int N);

#endif