#include "util.h"
#include "calc_error.h"

__device__ unsigned int log2(unsigned int n) {
    return (n > 1) ? 1 + log2(n >> 1) : 0;
}

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

__device__ unsigned long long zerosFP64 = 0;
__device__ unsigned long long zerosFP32 = 0;
__device__ unsigned long long negatives = 0;

__device__ unsigned long long zerosDiffGTNonZerosThresh = 0;
__device__ unsigned int maxDiffZerosDoubleVal = 0;
__device__ unsigned int maxDiffZerosFloatVal = 0;

__device__ unsigned long long diffDistribution[33] = {0};

unsigned long long get_zeros_fp64() {
    unsigned long long ret = 0;
    cudaMemcpyFromSymbol(&ret, zerosFP64, sizeof(unsigned long long), 0, cudaMemcpyDeviceToHost);
    return ret;
}

unsigned long long get_zeros_fp32() {
    unsigned long long ret = 0;
    cudaMemcpyFromSymbol(&ret, zerosFP32, sizeof(unsigned long long), 0, cudaMemcpyDeviceToHost);
    return ret;
}

unsigned long long get_negatives() {
    unsigned long long ret = 0;
    cudaMemcpyFromSymbol(&ret, negatives, sizeof(unsigned long long), 0, cudaMemcpyDeviceToHost);
    return ret;
}

unsigned long long get_zeros_diff_gt_non_zeros_thresh() {
    unsigned long long ret = 0;
    cudaMemcpyFromSymbol(&ret, zerosDiffGTNonZerosThresh, sizeof(unsigned long long), 0, cudaMemcpyDeviceToHost);
    return ret;
}

unsigned int get_max_diff_zeros_double_val() {
    unsigned int ret = 0;
    cudaMemcpyFromSymbol(&ret, maxDiffZerosDoubleVal, sizeof(unsigned int), 0, cudaMemcpyDeviceToHost);
    return ret;
}

unsigned int get_max_diff_zeros_float_val() {
    unsigned int ret = 0;
    cudaMemcpyFromSymbol(&ret, maxDiffZerosFloatVal, sizeof(unsigned int), 0, cudaMemcpyDeviceToHost);
    return ret;
}

void get_diffs_distribution(unsigned long long *dist) {
    cudaMemcpyFromSymbol(dist, diffDistribution, sizeof(unsigned long long) * 33, 0, cudaMemcpyDeviceToHost);
}

__forceinline__  __device__ void max_uint_error(double rhs, float lhs) {
    rhs = abs(rhs);
    lhs = abs(lhs);

    float rhs_as_float = __double2float_rz(rhs);
	uint32_t lhs_data = *((uint32_t*) &lhs);
    uint32_t rhs_data = *((uint32_t*) &rhs_as_float);

    uint32_t diff = SUB_ABS(lhs_data, rhs_data);

    int bit = __float2int_rd(log2(diff));
    atomicAdd(&(diffDistribution[bit]), 1);

    if (rhs == 0 || lhs == 0) {
        atomicMax(&maxUintErrorZeros, diff);
        if (diff > 30146560) {
            atomicAdd(&zerosDiffGTNonZerosThresh, 1);
            maxDiffZerosDoubleVal = rhs_data;
            maxDiffZerosFloatVal = lhs_data;
        }
    } else {
        atomicMax(&maxUintErrorNonZeros, diff);
    }
    
    if (rhs_as_float < 0 || lhs < 0) atomicAdd(&negatives, 1);
    if (rhs_as_float == 0) atomicAdd(&zerosFP64, 1);
    if (lhs == 0) atomicAdd(&zerosFP32, 1);
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

// > Max relative error

__device__ float relErrArray[1 << 24];
__device__ float absErrArray[1 << 24];

__global__ void find_max_relative_error_kernel(double *array, float *array_rp, int N) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N) {
        float lhs = array_rp[i];
        float rhs_as_float = __double2float_rz(array[i]);

        float rel_err = (rhs_as_float != 0) ? abs(1 - lhs / rhs_as_float) : -1;
        float abs_err = abs(lhs - rhs_as_float);

        relErrArray[i] = rel_err;
        absErrArray[i] = abs_err;
    }
}

float *tmpArr = (float*)calloc(sizeof(float), (1 << 24));

float maxRelErr = -9999;
double maxRelErrDoubleVal = 0;
float maxRelErrFloatVal = 0;
uint32_t maxRelErrUINTErr = 0;

float maxAbsErr = -9999;
double maxAbsErrDoubleVal = 0;
float maxAbsErrFloatVal = 0;
uint32_t maxAbsErrUINTErr = 0;

void find_max_relative_error_gpu(double *array, float *array_rp, int N) {
    int gridDim = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    find_max_relative_error_kernel<<<gridDim, BLOCK_SIZE>>>(array, array_rp, N);

    // > Relative error

    cudaMemcpyFromSymbol(tmpArr, relErrArray, sizeof(float) * N, 0, cudaMemcpyDeviceToHost);
    int max_rel_err_index = find_max_i(tmpArr, N);
    float max_rel_err = tmpArr[max_rel_err_index];

    double max_rel_err_double_val; cudaMemcpy(&max_rel_err_double_val, array + max_rel_err_index, sizeof(double), cudaMemcpyDeviceToHost);
    float max_rel_err_float_val; cudaMemcpy(&max_rel_err_float_val, array_rp + max_rel_err_index, sizeof(float), cudaMemcpyDeviceToHost);
    
    float rhs_as_float = (float)(max_rel_err_double_val);
	uint32_t lhs_data = *((uint32_t*) &max_rel_err_float_val);
    uint32_t rhs_data = *((uint32_t*) &rhs_as_float);
    uint32_t uintErr = SUB_ABS(lhs_data, rhs_data);

    if (max_rel_err > maxRelErr) {
        maxRelErr = max_rel_err;
        maxRelErrDoubleVal = max_rel_err_double_val;
        maxRelErrFloatVal = max_rel_err_float_val;
        maxRelErrUINTErr = uintErr;
    }

    // > Absolute error

    cudaMemcpyFromSymbol(tmpArr, absErrArray, sizeof(float) * N, 0, cudaMemcpyDeviceToHost);
    int max_abs_err_index = find_max_i(tmpArr, N);
    float max_abs_err = tmpArr[max_abs_err_index];

    double max_abs_err_double_val; cudaMemcpy(&max_abs_err_double_val, array + max_abs_err_index, sizeof(double), cudaMemcpyDeviceToHost);
    float max_abs_err_float_val; cudaMemcpy(&max_abs_err_float_val, array_rp + max_abs_err_index, sizeof(float), cudaMemcpyDeviceToHost);

    rhs_as_float = (float)(max_abs_err_double_val);
	lhs_data = *((uint32_t*) &max_abs_err_float_val);
    rhs_data = *((uint32_t*) &rhs_as_float);
    uintErr = SUB_ABS(lhs_data, rhs_data);

    if (max_abs_err > maxAbsErr) {
        maxAbsErr = max_abs_err;
        maxAbsErrDoubleVal = max_abs_err_double_val;
        maxAbsErrFloatVal = max_abs_err_float_val;
        maxAbsErrUINTErr = uintErr;
    }
}

float get_max_rel_error() { return maxRelErr; }
float get_max_rel_error_double_val() { return maxRelErrDoubleVal; }
float get_max_rel_error_float_val() { return maxRelErrFloatVal; }
float get_max_rel_error_uint_err() { return maxRelErrUINTErr; }

float get_max_abs_error() { return maxAbsErr; }
float get_max_abs_error_double_val() { return maxAbsErrDoubleVal; }
float get_max_abs_error_float_val() { return maxAbsErrFloatVal; }
float get_max_abs_error_uint_err() { return maxAbsErrUINTErr; }