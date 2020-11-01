#ifndef UTIL_H
#define UTIL_H

#include <stdio.h>
#include <time.h>
#include <sys/time.h>

#include <vector>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>

#define BLOCK_SIZE 32

#define CHECK_CUDA_ERROR(ans) { checkCudaError((ans), __FILE__, __LINE__); }
inline void checkCudaError(cudaError_t code, const char *file, int line, bool abort=false) {
    cudaDeviceSynchronize();
    if (code != cudaSuccess) {
        printf("CUDA Error: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

typedef struct timeval Time;
void getTimeNow(Time *t);
double elapsedTime(Time t1, Time t2);

int find_int_arg(int argc, char **argv, char *arg, int def);
char *find_char_arg(int argc, char **argv, char *arg, char *def);
float find_max(float *array, int N);
bool save_input(double *data, int dataN, double *kernel, int kernelN, float maxErr);
bool load_input(char *filename, double *data, int dataN, double *kernel, int kernelN);
bool save_output(double *output, int N, float maxErr);
bool save_output(double *output, int N);
bool compare_output_with_golden(double *output, int N, const char *filename);

#endif