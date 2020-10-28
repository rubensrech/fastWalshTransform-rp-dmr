/*
 * Copyright 1993-2007 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO USER:
 *
 * This source code is subject to NVIDIA ownership rights under U.S. and
 * international Copyright laws.  Users and possessors of this source code
 * are hereby granted a nonexclusive, royalty-free license to use this code
 * in individual and commercial software.
 *
 * NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE
 * CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR
 * IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH
 * REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL,
 * OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS
 * OF USE, DATA OR PROFITS,  WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
 * OR OTHER TORTIOUS ACTION,  ARISING OUT OF OR IN CONNECTION WITH THE USE
 * OR PERFORMANCE OF THIS SOURCE CODE.
 *
 * U.S. Government End Users.   This source code is a "commercial item" as
 * that term is defined at  48 C.F.R. 2.101 (OCT 1995), consisting  of
 * "commercial computer  software"  and "commercial computer software
 * documentation" as such terms are  used in 48 C.F.R. 12.212 (SEPT 1995)
 * and is provided to the U.S. Government only as a commercial end item.
 * Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through
 * 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the
 * source code with only those rights set forth herein.
 *
 * Any use of this source code in individual and commercial software must
 * include, in the user documentation and internal comments to the code,
 * the above Disclaimer and U.S. Government End Users Notice.
 */

#ifndef FWT_KERNEL_CU
#define FWT_KERNEL_CU

#include "util.h"
#include "relative_error.h"

///////////////////////////////////////////////////////////////////////////////
// Elementary(for vectors less than elementary size) in-shared memory 
// combined radix-2 + radix-4 Fast Walsh Transform 
///////////////////////////////////////////////////////////////////////////////
#define ELEMENTARY_LOG2SIZE 11

int iErr = 0;

__global__ void fwtBatch1Kernel(double *d_Output, float *d_Output_rp, double *d_Input, int log2N) {
    const int N = 1 << log2N;
    int stride = N;
    const int base = blockIdx.x << log2N;

    // (2 ** 11) * 4 bytes == 8KB -- maximum s_data[] size for G80
    extern __shared__ char s_data[];
    double *data    = (double*)s_data;
    float  *data_rp = (float*)(s_data + N*sizeof(double));

    double *d_Src = d_Input  + base;
    double *d_Dst = d_Output + base;                float *d_Dst_rp = d_Output_rp + base;

    for(int pos = threadIdx.x; pos < N; pos += blockDim.x) {
        data[pos] = d_Src[pos];                     data_rp[pos] = float(d_Src[pos]);
    }

    //Do single radix-2 stage if for odd power
    if(log2N & 1){
        __syncthreads();
        stride >>= 1;
        for(int pos = threadIdx.x; pos < N / 2; pos += blockDim.x){
            int lo = pos & (stride - 1);
            int i0 = ((pos - lo) << 1) + lo;
            int i1 = i0 + stride;

            double t0 = data[i0];                   float t0_rp = data_rp[i0];
            double t1 = data[i1];                   float t1_rp = data_rp[i1];
            data[i0] = t0 + t1;                     data_rp[i0] = t0_rp + t1_rp;
            data[i1] = t0 - t1;                     data_rp[i1] = t0_rp - t1_rp;
        }
    }

    //Main radix4 stages
    stride >>= 2;
    int pos = threadIdx.x;
    for(; stride >= 1; stride >>= 2){
        __syncthreads();
        int lo = pos & (stride - 1);
        int i0 = ((pos - lo) << 2) + lo;
        int i1 = i0 + stride;
        int i2 = i1 + stride;
        int i3 = i2 + stride;

        double d0 = data[i0];                       float d0_rp = data_rp[i0];
        double d1 = data[i1];                       float d1_rp = data_rp[i1];
        double d2 = data[i2];                       float d2_rp = data_rp[i2];
        double d3 = data[i3];                       float d3_rp = data_rp[i3];

        double t;
        t = d0; d0         = d0 + d2; d2         = t - d2;
        t = d1; d1         = d1 + d3; d3         = t - d3;
        t = d0; data[i0] = d0 + d1; data[i1] = t - d1;
        t = d2; data[i2] = d2 + d3; data[i3] = t - d3;

        float t_rp;
        t_rp = d0_rp; d0_rp         = d0_rp + d2_rp; d2_rp         = t_rp - d2_rp;
        t_rp = d1_rp; d1_rp         = d1_rp + d3_rp; d3_rp         = t_rp - d3_rp;
        t_rp = d0_rp; data_rp[i0] = d0 + d1; data_rp[i1] = t_rp - d1_rp;
        t_rp = d2_rp; data_rp[i2] = d2 + d3; data_rp[i3] = t_rp - d3_rp;
    }

    __syncthreads();
    for(int pos = threadIdx.x; pos < N; pos += blockDim.x) {
        d_Dst[pos] = data[pos];                     d_Dst_rp[pos] = data_rp[pos];
    }
}

////////////////////////////////////////////////////////////////////////////////
// Single in-global memory radix-4 Fast Walsh Transform pass
// (for strides exceeding elementary vector size)
////////////////////////////////////////////////////////////////////////////////
__global__ void fwtBatch2Kernel(double *d_Output, float *d_Output_rp, double *d_Input, int stride) {
    const int pos = blockIdx.x * blockDim.x + threadIdx.x;
    const int   N = blockDim.x *  gridDim.x * 4;

    double *d_Src = d_Input   + blockIdx.y * N;
    double *d_Dst = d_Output + blockIdx.y * N;      float *d_Dst_rp = d_Output_rp + blockIdx.y * N;

    int lo = pos & (stride - 1);
    int i0 = ((pos - lo) << 2) + lo;
    int i1 = i0 + stride;
    int i2 = i1 + stride;
    int i3 = i2 + stride;

    double d0 = d_Src[i0];                          float d0_rp = float(d_Src[i0]);
    double d1 = d_Src[i1];                          float d1_rp = float(d_Src[i1]);
    double d2 = d_Src[i2];                          float d2_rp = float(d_Src[i2]);
    double d3 = d_Src[i3];                          float d3_rp = float(d_Src[i3]);

    double t;
    t = d0; d0        = d0 + d2; d2        = t - d2;
    t = d1; d1        = d1 + d3; d3        = t - d3;
    t = d0; d_Dst[i0] = d0 + d1; d_Dst[i1] = t - d1;
    t = d2; d_Dst[i2] = d2 + d3; d_Dst[i3] = t - d3;

    float t_rp;
    t_rp = d0_rp; d0_rp        = d0_rp + d2_rp; d2_rp        = t_rp - d2_rp;
    t_rp = d1_rp; d1_rp        = d1_rp + d3_rp; d3_rp        = t_rp - d3_rp;
    t_rp = d0_rp; d_Dst_rp[i0] = d0_rp + d1_rp; d_Dst_rp[i1] = t_rp - d1_rp;
    t_rp = d2_rp; d_Dst_rp[i2] = d2_rp + d3_rp; d_Dst_rp[i3] = t_rp - d3_rp;
}

////////////////////////////////////////////////////////////////////////////////
// Put everything together: batched Fast Walsh Transform CPU front-end
////////////////////////////////////////////////////////////////////////////////
void fwtBatchGPU(double *d_Data, float *d_Output_rp, int M, int log2N, float *h_err, float *d_err, std::vector<float> &max_errs) {
    int N = 1 << log2N;
    dim3 grid((1 << log2N) / 1024, M, 1);
    for(; log2N > ELEMENTARY_LOG2SIZE; log2N -= 2, N >>= 2, M <<= 2){
        fwtBatch2Kernel<<<grid, 256>>>(d_Data, d_Output_rp, d_Data, N / 4);
        CHECK_CUDA_ERROR(cudaPeekAtLastError());
        check_relative_error_gpu(d_Data, d_Output_rp, N);
        
        calc_relative_error_gpu(d_Data, d_Output_rp, d_err, N);
        cudaMemcpy(h_err, d_err, N * sizeof(float), cudaMemcpyDeviceToHost);
        max_errs.push_back(find_max(h_err, N));
    }

    fwtBatch1Kernel<<<M, N / 4, N * sizeof(double) + N * sizeof(float)>>>(d_Data, d_Output_rp, d_Data, log2N);
    CHECK_CUDA_ERROR(cudaPeekAtLastError());
    check_relative_error_gpu(d_Data, d_Output_rp, N);

    calc_relative_error_gpu(d_Data, d_Output_rp, d_err, N);
    cudaMemcpy(h_err, d_err, N * sizeof(float), cudaMemcpyDeviceToHost);
    max_errs[iErr++] = find_max(h_err, N);
}

////////////////////////////////////////////////////////////////////////////////
// Modulate two arrays
////////////////////////////////////////////////////////////////////////////////
__global__ void modulateKernel(double *d_A, float *d_A_rp, double *d_B, int N){
    int        tid = blockIdx.x * blockDim.x + threadIdx.x;
    int numThreads = blockDim.x * gridDim.x;
    double     rcpN = 1.0f / (double)N;             float rcpN_rp = 1.0f / (float)N;

    for (int pos = tid; pos < N; pos += numThreads) {
        d_A[pos] *= d_B[pos] * rcpN;                d_A_rp[pos] *= float(d_B[pos]) * rcpN_rp;
    }
}

// Interface to modulateKernel()
void modulateGPU(double *d_A, float *d_A_rp, double *d_B, int N, float *h_err, float *d_err, std::vector<float> &max_errs) {
    modulateKernel<<<128, 256>>>(d_A, d_A_rp, d_B, N);
    check_relative_error_gpu(d_A, d_A_rp, N);

    calc_relative_error_gpu(d_A, d_A_rp, d_err, N);
    cudaMemcpy(h_err, d_err, N * sizeof(float), cudaMemcpyDeviceToHost);
    max_errs.push_back(find_max(h_err, N));
}

////////////////////////////////////////////////////////////////////////////////
// Extra functions
////////////////////////////////////////////////////////////////////////////////

__global__ void copyKernel(float *array_rp, double *array, int N) {
    int tid =  blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N)
        array_rp[tid] = array[tid];
}

void copyGPU(float *array_rp, double *array, int N) {
    int blockSize = 32;
    int gridSize = (N + blockSize - 1) / blockSize;
    copyKernel<<<gridSize, blockSize>>>(array_rp, array, N);
    CHECK_CUDA_ERROR(cudaPeekAtLastError());
}

#endif
