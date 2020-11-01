#include "util.h"
#include "relative_error.h"

////////////////////////////////////////////////////////////////////////////////
// Relative error functions
////////////////////////////////////////////////////////////////////////////////

// > Check relative error

__device__ unsigned long long errors = 0;

unsigned long long get_dmr_error() {
    unsigned long long ret = 0;
    cudaMemcpyFromSymbol(&ret, errors, sizeof(unsigned long long), 0, cudaMemcpyDeviceToHost);
    return ret;
}

__forceinline__  __device__ void relative_error(double val, float val_rp) {
    float relative = __fdividef(val_rp, float(val));
    if (val > 0 && (relative < MIN_PERCENTAGE || relative > MAX_PERCENTAGE)) {
        atomicAdd(&errors, 1);
    }
}

__global__ void check_relative_error_kernel(double *array, float *array_rp, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N)
        relative_error(array[tid], array_rp[tid]);
}

void check_relative_error_gpu(double *array, float *array_rp, int N) {
    int gridDim = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    check_relative_error_kernel<<<gridDim, BLOCK_SIZE>>>(array, array_rp, N);
    CHECK_CUDA_ERROR(cudaPeekAtLastError());
}

// > Calc relative error

__global__ void calc_relative_error_kernel(double *array, float *array_rp, float *err_out, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
        if (array[tid] > 0) {
            err_out[tid] = abs(1 - __fdividef(array_rp[tid], float(array[tid])));
        } else {
            err_out[tid] = 0;
        }   
    }
}

void calc_relative_error_gpu(double *array, float *array_rp, float *err_out, int N) {
    int gridDim = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    calc_relative_error_kernel<<<gridDim, BLOCK_SIZE>>>(array, array_rp, err_out, N);
    CHECK_CUDA_ERROR(cudaPeekAtLastError());
}

// > Max relative error

float find_max_relative_error_gpu(double *d_array, float *d_array_rp, int N, float *h_err, float *d_err) {
    calc_relative_error_gpu(d_array, d_array_rp, d_err, N);
    cudaMemcpy(h_err, d_err, N * sizeof(float), cudaMemcpyDeviceToHost);
    return find_max(h_err, N);
}