#include "util.h"
#include "calc_error.h"

__device__ unsigned int log2(unsigned int n) {
    return (n > 1) ? 1 + log2(n >> 1) : 0;
}

////////////////////////////////////////////////////////////////////////////////
// Check error functions
////////////////////////////////////////////////////////////////////////////////

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
    if (relative < MIN_PERCENTAGE || relative > MAX_PERCENTAGE) {
        atomicAdd(&errors, 1);
    }
}

__forceinline__  __device__ void uint_error(double rhs, float lhs) {
	float rhs_as_float = float(rhs);
	uint32_t lhs_data = *((uint32_t*) &lhs);
	uint32_t rhs_data = *((uint32_t*) &rhs_as_float);

	uint32_t diff = SUB_ABS(lhs_data, rhs_data);

	if (diff > UINT_THRESHOLD) {
		atomicAdd(&errors, 1);
	}
}

__forceinline__  __device__ void hybrid_error(double val, float val_rp) {
    float lhs = abs(val_rp);
    float rhs = __double2float_rz(abs(val));

    if (rhs == 0 || lhs == 0) {
        // ABSOLUTE ERROR
        float abs_err = SUB_ABS(rhs, lhs);
        if (abs_err > ABS_ERR_THRESHOLD) {
            atomicAdd(&errors, 1);
        }
    } else if (rhs < ABS_ERR_UPPER_BOUND_VAL && lhs < ABS_ERR_UPPER_BOUND_VAL) {
        // ABSOLUTE ERROR
        float abs_err = SUB_ABS(rhs, lhs);
        if (abs_err > ABS_ERR_THRESHOLD) {
            atomicAdd(&errors, 1);
        }
    } else if (rhs >= ABS_ERR_UPPER_BOUND_VAL || lhs >= ABS_ERR_UPPER_BOUND_VAL) {
        // RELATIVE ERROR
        float rel_err = SUB_ABS(1, lhs / rhs);
        if (rel_err > REL_ERR_THRESHOLD) {
            atomicAdd(&errors, 1);
        }
    }
}

__global__ void check_errors_kernel(double *array, float *array_rp, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N)
#if ERROR_METRIC == HYBRID
        hybrid_error(array[tid], array_rp[tid]);
#elif ERROR_METRIC == UINT_ERROR
        uint_error(array[tid], array_rp[tid]);
#else
        relative_error(array[tid], array_rp[tid]);
#endif
}

void check_errors_gpu(double *array, float *array_rp, int N) {
    int gridDim = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    check_errors_kernel<<<gridDim, BLOCK_SIZE>>>(array, array_rp, N);
    CHECK_CUDA_ERROR(cudaPeekAtLastError());
}

////////////////////////////////////////////////////////////////////////////////
// Find max error functions
////////////////////////////////////////////////////////////////////////////////

__device__ unsigned long long uintErrDistribution[32] = {0};
__device__ uint32_t maxUintError = 0;

__device__ float relErrArray[1 << 24];

__device__ float absErrArray[1 << 24];

__device__ uint32_t zerosFP64 = 0;
__device__ uint32_t zerosFP32 = 0;
__device__ uint32_t negatives = 0;

__global__ void calc_errors_kernel(double *array, float *array_rp, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    double rhs = array[i];
    float lhs = array_rp[i];
    float rhs_as_float = __double2float_rz(rhs);

    // > UINT error
	uint32_t lhs_data = *((uint32_t*) &lhs);
    uint32_t rhs_data = *((uint32_t*) &rhs_as_float);
    uint32_t diff = SUB_ABS(lhs_data, rhs_data);

    int bit = __float2int_rd(log2(diff));
    atomicAdd(&(uintErrDistribution[bit]), 1);

    atomicMax(&maxUintError, diff);

    // > Relative error
    relErrArray[i] = (rhs_as_float != 0) ? abs(1 - lhs / rhs_as_float) : IGNORE_VAL_FLAG;

    // > Absolute error
    absErrArray[i] = abs(lhs - rhs_as_float);
    
    // > Stats
    if (rhs_as_float < 0 || lhs < 0) atomicAdd(&negatives, 1);
    if (rhs_as_float == 0) atomicAdd(&zerosFP64, 1);
    if (lhs == 0) atomicAdd(&zerosFP32, 1);
}

void calc_errors_gpu(double *array, float *array_rp, int N) {
    int gridDim = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    calc_errors_kernel<<<gridDim, BLOCK_SIZE>>>(array, array_rp, N);
    CHECK_CUDA_ERROR(cudaPeekAtLastError());
}

// > Getters

void get_diffs_distribution(unsigned long long *dist) {
    cudaMemcpyFromSymbol(dist, uintErrDistribution, sizeof(unsigned long long) * 33, 0, cudaMemcpyDeviceToHost);
}

uint32_t get_max_uint_err() {
    uint32_t ret = 0;
    cudaMemcpyFromSymbol(&ret, maxUintError, sizeof(uint32_t), 0, cudaMemcpyDeviceToHost);
    return ret;
}

void get_rel_error_array(float *relErrArr, int N) {
    cudaMemcpyFromSymbol(relErrArr, relErrArray, sizeof(float) * N, 0, cudaMemcpyDeviceToHost);
}

void get_abs_error_array(float *absErrArr, int N) {
    cudaMemcpyFromSymbol(absErrArr, absErrArray, sizeof(float) * N, 0, cudaMemcpyDeviceToHost);
}

uint32_t get_zeros_fp64() {
    uint32_t ret = 0;
    cudaMemcpyFromSymbol(&ret, zerosFP64, sizeof(uint32_t), 0, cudaMemcpyDeviceToHost);
    return ret;
}

uint32_t get_zeros_fp32() {
    uint32_t ret = 0;
    cudaMemcpyFromSymbol(&ret, zerosFP32, sizeof(uint32_t), 0, cudaMemcpyDeviceToHost);
    return ret;
}

uint32_t get_negatives() {
    uint32_t ret = 0;
    cudaMemcpyFromSymbol(&ret, negatives, sizeof(uint32_t), 0, cudaMemcpyDeviceToHost);
    return ret;
}


// > Hybrid error (relative + abs error)

// __global__ void calc_error_hybrid_kernel(double *array, float *array_rp, int N) {
//     int i = blockDim.x * blockIdx.x + threadIdx.x;
//     if (i < N) {
//         float lhs = abs(array_rp[i]);
//         float rhs = __double2float_rz(abs(array[i]));

//         if (rhs == 0 || lhs == 0) {
//             // ABSOLUTE ERROR
//             absErrArray[i] = abs(rhs - lhs);
//             relErrArray[i] = IGNORE_VAL_FLAG;
//         } else if (rhs < ABS_ERR_UPPER_BOUND_VAL && lhs < ABS_ERR_UPPER_BOUND_VAL) {
//             // ABSOLUTE ERROR
//             absErrArray[i] = abs(rhs - lhs);
//             relErrArray[i] = IGNORE_VAL_FLAG;
//         } else if (rhs >= ABS_ERR_UPPER_BOUND_VAL || lhs >= ABS_ERR_UPPER_BOUND_VAL) {
//             // RELATIVE ERROR
//             absErrArray[i] = IGNORE_VAL_FLAG;
//             relErrArray[i] = abs(1 - lhs / rhs);
//         }
//     }
// }

// void calc_error_hybrid_gpu(double *array, float *array_rp, int N) {
//     int gridDim = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
//     calc_error_hybrid_kernel<<<gridDim, BLOCK_SIZE>>>(array, array_rp, N);

//     // > Relative error

//     cudaMemcpyFromSymbol(tmpArr, relErrArray, sizeof(float) * N, 0, cudaMemcpyDeviceToHost);
//     int max_rel_err_index = find_max_i(tmpArr, N);
//     float max_rel_err = tmpArr[max_rel_err_index];

//     double max_rel_err_double_val; cudaMemcpy(&max_rel_err_double_val, array + max_rel_err_index, sizeof(double), cudaMemcpyDeviceToHost);
//     float max_rel_err_float_val; cudaMemcpy(&max_rel_err_float_val, array_rp + max_rel_err_index, sizeof(float), cudaMemcpyDeviceToHost);
    
//     float rhs_as_float = (float)(max_rel_err_double_val);
// 	uint32_t lhs_data = *((uint32_t*) &max_rel_err_float_val);
//     uint32_t rhs_data = *((uint32_t*) &rhs_as_float);
//     uint32_t uintErr = SUB_ABS(lhs_data, rhs_data);

//     if (max_rel_err > maxRelErr) {
//         maxRelErr = max_rel_err;
//         maxRelErrDoubleVal = max_rel_err_double_val;
//         maxRelErrFloatVal = max_rel_err_float_val;
//         maxRelErrUINTErr = uintErr;
//     }

//     // > Absolute error

//     cudaMemcpyFromSymbol(tmpArr, absErrArray, sizeof(float) * N, 0, cudaMemcpyDeviceToHost);
//     int max_abs_err_index = find_max_i(tmpArr, N);
//     float max_abs_err = tmpArr[max_abs_err_index];

//     double max_abs_err_double_val; cudaMemcpy(&max_abs_err_double_val, array + max_abs_err_index, sizeof(double), cudaMemcpyDeviceToHost);
//     float max_abs_err_float_val; cudaMemcpy(&max_abs_err_float_val, array_rp + max_abs_err_index, sizeof(float), cudaMemcpyDeviceToHost);

//     rhs_as_float = (float)(max_abs_err_double_val);
// 	lhs_data = *((uint32_t*) &max_abs_err_float_val);
//     rhs_data = *((uint32_t*) &rhs_as_float);
//     uintErr = SUB_ABS(lhs_data, rhs_data);

//     if (max_abs_err > maxAbsErr) {
//         maxAbsErr = max_abs_err;
//         maxAbsErrDoubleVal = max_abs_err_double_val;
//         maxAbsErrFloatVal = max_abs_err_float_val;
//         maxAbsErrUINTErr = uintErr;
//     }
// }