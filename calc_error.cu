#include "util.h"
#include "calc_error.h"

////////////////////////////////////////////////////////////////////////////////
// Relative error functions
////////////////////////////////////////////////////////////////////////////////

// > Check relative error

__device__ unsigned long long errors = 0;

__device__ unsigned int maxUintErrorNonZeros = 0;
__device__ unsigned int maxUintErrorZeros = 0;

unsigned long long get_dmr_error() {
    unsigned long long ret = 0;
    cudaMemcpyFromSymbol(&ret, errors, sizeof(unsigned long long), 0, cudaMemcpyDeviceToHost);
    return ret;
}

__forceinline__  __device__ void relative_error(double val, float val_rp) {
    float relative = __fdividef(val_rp, float(val));
    if (val == 0) {
        if (val != val_rp) {
            atomicAdd(&errors, 1);
        }
    } else if (relative < MIN_PERCENTAGE || relative > MAX_PERCENTAGE) {
        atomicAdd(&errors, 1);
    }
}

__forceinline__  __device__ void uint_error(double rhs, float lhs, uint32_t threshold) {
	float rhs_as_float = float(rhs);
	uint32_t lhs_data = *((uint32_t*) &lhs);
	uint32_t rhs_data = *((uint32_t*) &rhs_as_float);

	uint32_t diff = SUB_ABS(lhs_data, rhs_data);

	if (diff > threshold) {
		atomicAdd(&errors, 1);
	}
}

__global__ void check_error_kernel(double *array, float *array_rp, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N)
#if ERROR_METRIC == UINT_ERROR
        uint_error(array[tid], array_rp[tid], UINT_THRESHOLD);
#else
        relative_error(array[tid], array_rp[tid]);
#endif
}

void check_error_gpu(double *array, float *array_rp, int N) {
    int gridDim = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    check_error_kernel<<<gridDim, BLOCK_SIZE>>>(array, array_rp, N);
    CHECK_CUDA_ERROR(cudaPeekAtLastError());
}

// > Max UINT error

__forceinline__  __device__ void max_uint_error(double rhs, float lhs) {
	float rhs_as_float = float(rhs);
	uint32_t lhs_data = *((uint32_t*) &lhs);
	uint32_t rhs_data = *((uint32_t*) &rhs_as_float);

	uint32_t diff = SUB_ABS(lhs_data, rhs_data);

    if (rhs == 0 || lhs == 0) atomicMax(&maxUintErrorZeros, diff);
    else                      atomicMax(&maxUintErrorNonZeros, diff);
}

__global__ void find_max_uint_error_kernel(double *array, float *array_rp, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N)
        max_uint_error(array[tid], array_rp[tid]);
}

void find_max_uint_error_gpu(double *array, float *array_rp, int N) {
    int gridDim = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    find_max_uint_error_kernel<<<gridDim, BLOCK_SIZE>>>(array, array_rp, N);
    CHECK_CUDA_ERROR(cudaPeekAtLastError());
}

unsigned int get_max_uint_error_non_zeros() {
    unsigned int ret = 0;
    cudaMemcpyFromSymbol(&ret, maxUintErrorNonZeros, sizeof(unsigned int), 0, cudaMemcpyDeviceToHost);
    return ret;
}

unsigned int get_max_uint_error_zeros() {
    unsigned int ret = 0;
    cudaMemcpyFromSymbol(&ret, maxUintErrorZeros, sizeof(unsigned int), 0, cudaMemcpyDeviceToHost);
    return ret;
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
