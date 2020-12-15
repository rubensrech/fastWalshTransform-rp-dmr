#ifndef UTIL_H
#define UTIL_H

#include <stdio.h>
#include <time.h>
#include <sys/time.h>

#include <vector>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>

#define SUB_ABS(lhs, rhs) ((lhs > rhs) ? (lhs - rhs) : (rhs - lhs))

#define BLACK       "\e[0;30m"
#define RED         "\e[0;31m"
#define GREEN       "\e[0;32m"
#define YELLOW      "\e[0;33m"
#define BLUE        "\e[0;34m"
#define PURPLE      "\e[0;35m"
#define CYAN        "\e[0;36m"
#define WHITE       "\e[0;37m"
#define DFT_COLOR   "\e[0m"

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

unsigned int log2_host(unsigned int n);

int find_int_arg(int argc, char **argv, char *arg, int def);
char *find_char_arg(int argc, char **argv, char *arg, char *def);
float find_max_i(float *array, int N);
bool save_input(double *data, int dataN, double *kernel, int kernelN, float maxErr);
bool save_input(double *data, int dataN, double *kernel, int kernelN, int maxErrBit);
bool load_input(char *filename, double *data, int dataN, double *kernel, int kernelN);
bool save_output(double *output, int N, float maxErr);
bool save_output(double *output, int N);
bool compare_output_with_golden(double *output, int N, const char *filename);

#endif