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

///////////////////////////////////////////////////////////////////////////////
// Elementary(for vectors less than elementary size) in-shared memory 
// combined radix-2 + radix-4 Fast Walsh Transform 
///////////////////////////////////////////////////////////////////////////////
#define ELEMENTARY_LOG2SIZE 11

__global__ void fwtBatch1Kernel(double *d_Output, double *d_Input, int log2N) {
    const int N = 1 << log2N;
    int stride = N;
    const int base = blockIdx.x << log2N;

    // (2 ** 11) * 4 bytes == 8KB -- maximum s_data[] size for G80
    extern __shared__ double s_data[];
    double *d_Src = d_Input  + base;
    double *d_Dst = d_Output + base;

    for(int pos = threadIdx.x; pos < N; pos += blockDim.x)
        s_data[pos] = d_Src[pos];

    //Do single radix-2 stage if for odd power
    if(log2N & 1){
        __syncthreads();
        stride >>= 1;
        for(int pos = threadIdx.x; pos < N / 2; pos += blockDim.x){
            int lo = pos & (stride - 1);
            int i0 = ((pos - lo) << 1) + lo;
            int i1 = i0 + stride;

            double t0 = s_data[i0];
            double t1 = s_data[i1];
            s_data[i0] = t0 + t1;
            s_data[i1] = t0 - t1;
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

        double d0 = s_data[i0];
        double d1 = s_data[i1];
        double d2 = s_data[i2];
        double d3 = s_data[i3];

        double t;
        t = d0; d0         = d0 + d2; d2         = t - d2;
        t = d1; d1         = d1 + d3; d3         = t - d3;
        t = d0; s_data[i0] = d0 + d1; s_data[i1] = t - d1;
        t = d2; s_data[i2] = d2 + d3; s_data[i3] = t - d3;
    }

    __syncthreads();
    for(int pos = threadIdx.x; pos < N; pos += blockDim.x)
        d_Dst[pos] = s_data[pos];
}

__global__ void fwtBatch1Kernel(float *d_Output, float *d_Input, int log2N) {
    const int N = 1 << log2N;
    int stride = N;
    const int base = blockIdx.x << log2N;

    // (2 ** 11) * 4 bytes == 8KB -- maximum s_data[] size for G80
    extern __shared__ float s_data_rp[];
    float *d_Src = d_Input  + base;
    float *d_Dst = d_Output + base;

    for(int pos = threadIdx.x; pos < N; pos += blockDim.x)
        s_data_rp[pos] = d_Src[pos];

    //Do single radix-2 stage if for odd power
    if(log2N & 1){
        __syncthreads();
        stride >>= 1;
        for(int pos = threadIdx.x; pos < N / 2; pos += blockDim.x){
            int lo = pos & (stride - 1);
            int i0 = ((pos - lo) << 1) + lo;
            int i1 = i0 + stride;

            float t0 = s_data_rp[i0];
            float t1 = s_data_rp[i1];
            s_data_rp[i0] = t0 + t1;
            s_data_rp[i1] = t0 - t1;
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

        float d0 = s_data_rp[i0];
        float d1 = s_data_rp[i1];
        float d2 = s_data_rp[i2];
        float d3 = s_data_rp[i3];

        float t;
        t = d0; d0         = d0 + d2; d2         = t - d2;
        t = d1; d1         = d1 + d3; d3         = t - d3;
        t = d0; s_data_rp[i0] = d0 + d1; s_data_rp[i1] = t - d1;
        t = d2; s_data_rp[i2] = d2 + d3; s_data_rp[i3] = t - d3;
    }

    __syncthreads();
    for(int pos = threadIdx.x; pos < N; pos += blockDim.x)
        d_Dst[pos] = s_data_rp[pos];
}

////////////////////////////////////////////////////////////////////////////////
// Single in-global memory radix-4 Fast Walsh Transform pass
// (for strides exceeding elementary vector size)
////////////////////////////////////////////////////////////////////////////////
template<typename real_t>
__global__ void fwtBatch2Kernel(real_t *d_Output, real_t *d_Input, int stride) {
    const int pos = blockIdx.x * blockDim.x + threadIdx.x;
    const int   N = blockDim.x *  gridDim.x * 4;

    real_t *d_Src = d_Input  + blockIdx.y * N;
    real_t *d_Dst = d_Output + blockIdx.y * N;

    int lo = pos & (stride - 1);
    int i0 = ((pos - lo) << 2) + lo;
    int i1 = i0 + stride;
    int i2 = i1 + stride;
    int i3 = i2 + stride;

    real_t d0 = d_Src[i0];
    real_t d1 = d_Src[i1];
    real_t d2 = d_Src[i2];
    real_t d3 = d_Src[i3];

    real_t t;
    t = d0; d0        = d0 + d2; d2        = t - d2;
    t = d1; d1        = d1 + d3; d3        = t - d3;
    t = d0; d_Dst[i0] = d0 + d1; d_Dst[i1] = t - d1;
    t = d2; d_Dst[i2] = d2 + d3; d_Dst[i3] = t - d3;
}

////////////////////////////////////////////////////////////////////////////////
// Put everything together: batched Fast Walsh Transform CPU front-end
////////////////////////////////////////////////////////////////////////////////
template<typename real_t>
void fwtBatchGPUTemplate(real_t *d_Data, int M, int log2N, cudaStream_t stream) {
    int N = 1 << log2N;
    dim3 grid((1 << log2N) / 1024, M, 1);
    for(; log2N > ELEMENTARY_LOG2SIZE; log2N -= 2, N >>= 2, M <<= 2){
        fwtBatch2Kernel<<<grid, 256, 0, stream>>>(d_Data, d_Data, N / 4);
        CHECK_CUDA_ERROR(cudaPeekAtLastError());
    }

    fwtBatch1Kernel<<<M, N / 4, N * sizeof(real_t), stream>>>(d_Data, d_Data, log2N);
    CHECK_CUDA_ERROR(cudaPeekAtLastError());
}

void fwtBatchGPU(double *d_Data, int M, int log2N, cudaStream_t stream) {
    fwtBatchGPUTemplate(d_Data, M, log2N, stream);
}

void fwtBatchGPU(float *d_Data, int M, int log2N, cudaStream_t stream) {
    fwtBatchGPUTemplate(d_Data, M, log2N, stream);
}

////////////////////////////////////////////////////////////////////////////////
// Modulate two arrays
////////////////////////////////////////////////////////////////////////////////
template<typename real_t>
__global__ void modulateKernel(real_t *d_A, real_t *d_B, int N){
    int        tid = blockIdx.x * blockDim.x + threadIdx.x;
    int numThreads = blockDim.x * gridDim.x;
    real_t     rcpN = 1.0f / (real_t)N;

    for (int pos = tid; pos < N; pos += numThreads)
        d_A[pos] *= d_B[pos] * rcpN;
}

// Interface to modulateKernel()
void modulateGPU(double *d_A, double *d_B, int N, cudaStream_t stream) {
    modulateKernel<<<128, 256, 0, stream>>>(d_A, d_B, N);
}

void modulateGPU(float *d_A, float *d_B, int N, cudaStream_t stream) {
    modulateKernel<<<128, 256, 0, stream>>>(d_A, d_B, N);
}

#endif
