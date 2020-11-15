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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>

#include "util.h"

////////////////////////////////////////////////////////////////////////////////
// CPU FWT
////////////////////////////////////////////////////////////////////////////////
void fwtCPU(double *h_Output, double *h_Input, int log2N){
    const int N = 1 << log2N;

    for(int pos = 0; pos < N; pos++)
        h_Output[pos] = h_Input[pos];

    //Cycle through stages with different butterfly strides
    for(int stride = N / 2; stride >= 1; stride >>= 1){
        //Cycle through subvectors of (2 * stride) elements
        for(int base = 0; base < N; base += 2 * stride)
            //Butterfly index within subvector of (2 * stride) size
            for(int j = 0; j < stride; j++){
                int i0 = base + j +      0;
                int i1 = base + j + stride;

                double T1 = h_Output[i0];
                double T2 = h_Output[i1];
                h_Output[i0] = T1 + T2;
                h_Output[i1] = T1 - T2;
            }
    }
}

////////////////////////////////////////////////////////////////////////////////
// GPU FWT
////////////////////////////////////////////////////////////////////////////////
extern void fwtBatchGPU(double *d_Data, int M, int log2N, cudaStream_t stream);
extern void fwtBatchGPU(float *d_Data, int M, int log2N, cudaStream_t stream);
extern void modulateGPU(double *d_A, double *d_B, int N, cudaStream_t stream);
extern void modulateGPU(float *d_A, float *d_B, int N, cudaStream_t stream);

extern void relative_error_gpu(double *output, float *output_rp, float *err_output, int N);

////////////////////////////////////////////////////////////////////////////////
// Data configuration
////////////////////////////////////////////////////////////////////////////////
const int log2Kernel = 7;

#ifndef __DEVICE_EMULATION__
    const   int log2Data = 24;
#else
    const   int log2Data = 16;
#endif
const int dataN = 1 << log2Data;
const int kernelN = 1 << log2Kernel;

const int DATA_SIZE = dataN   * sizeof(double);
const int DATA_SIZE_RP = dataN   * sizeof(float);
const int KERNEL_SIZE = kernelN * sizeof(double);
const int KERNEL_SIZE_RP = kernelN * sizeof(float);

////////////////////////////////////////////////////////////////////////////////
// Timing functions
////////////////////////////////////////////////////////////////////////////////
typedef struct timeval Time;

void getTimeNow(Time *t) {
    gettimeofday(t, 0);
}

double elapsedTime(Time t1, Time t2) {
    return (1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec)/1000.0;
}

////////////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char *argv[]) {
    // Host data
    double *h_Data, *h_Kernel, *h_ResultCPU, *h_ResultGPU;
    float *h_Data_rp, *h_Kernel_rp, *h_ResultGPU_rp;
    float *h_Error;

    // Device data
    double *d_Data, *d_Kernel;
    float *d_Data_rp, *d_Kernel_rp;
    float *d_Error;
    cudaStream_t stream1;
    cudaEvent_t startStream1, stopStream1;

    double delta, ref, sum_delta2, sum_ref2, L2norm;
    Time t1, t2;
    int i;

    // ==========================================================================
    // ==========================================================================
    printf("1) Initializing data\n");

    cudaEventCreate(&startStream1);
    cudaEventCreate(&stopStream1);

    // ====================================================
    printf("    1.1) Allocating CPU memory... ");
    getTimeNow(&t1);

    // Full-precision
    h_Kernel    = (double*)malloc(KERNEL_SIZE);
    h_Data      = (double*)malloc(DATA_SIZE);
    h_ResultCPU = (double*)malloc(DATA_SIZE);
    h_ResultGPU = (double*)malloc(DATA_SIZE);
    // Reduced-precision
    h_Kernel_rp    = (float*)malloc(KERNEL_SIZE_RP);
    h_Data_rp      = (float*)malloc(DATA_SIZE_RP);
    h_ResultGPU_rp = (float*)malloc(DATA_SIZE_RP);
    // Error calculation
    h_Error = (float*)malloc(DATA_SIZE_RP);

    getTimeNow(&t2);
    printf("(%3.3lf ms)\n", elapsedTime(t1, t2));
    

    // ====================================================
    printf("    1.2) Allocating GPU memory... ");
    getTimeNow(&t1);

    CHECK_CUDA_ERROR(cudaStreamCreateWithFlags(&stream1, cudaStreamNonBlocking));
    // Full-precision
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_Kernel, DATA_SIZE));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_Data,   DATA_SIZE));
    // Reduced-precision
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_Kernel_rp, DATA_SIZE_RP));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_Data_rp,   DATA_SIZE_RP));
    // Error calculation
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_Error,   DATA_SIZE_RP));

    getTimeNow(&t2);
    printf("(%3.3lf ms)\n", elapsedTime(t1, t2));

    // ====================================================
    printf("    1.3) Generating data (host)... ");
    getTimeNow(&t1);

    srand((int)time(NULL));
    for (i = 0; i < kernelN; i++) {
        h_Kernel[i] = (double)rand() / (double)RAND_MAX;
        h_Kernel_rp[i] = h_Kernel[i];
    }
    for (i = 0; i < dataN; i++) {
        h_Data[i] = (double)rand() / (double)RAND_MAX;
        h_Data_rp[i] = h_Data[i];
    }

    getTimeNow(&t2);
    printf("(%3.3lf ms)\n", elapsedTime(t1, t2));

    // ====================================================
    printf("    1.4) Copying data to device... ");
    getTimeNow(&t1);

    // Full-precision    
    CHECK_CUDA_ERROR(cudaMemsetAsync(d_Kernel, 0, DATA_SIZE, stream1));
    CHECK_CUDA_ERROR(cudaMemcpyAsync(d_Kernel, h_Kernel, KERNEL_SIZE, cudaMemcpyHostToDevice, stream1));
    CHECK_CUDA_ERROR(cudaMemcpyAsync(d_Data, h_Data, DATA_SIZE, cudaMemcpyHostToDevice, stream1));
    // Reduced-precision
    CHECK_CUDA_ERROR(cudaMemsetAsync(d_Kernel_rp, 0, DATA_SIZE_RP, stream1));
    CHECK_CUDA_ERROR(cudaMemcpyAsync(d_Kernel_rp, h_Kernel_rp, KERNEL_SIZE_RP, cudaMemcpyHostToDevice, stream1));
    CHECK_CUDA_ERROR(cudaMemcpyAsync(d_Data_rp, h_Data_rp, DATA_SIZE_RP, cudaMemcpyHostToDevice, stream1));

    cudaStreamSynchronize(stream1);
    cudaDeviceSynchronize();
    getTimeNow(&t2);
    printf("(%3.3lf ms)\n", elapsedTime(t1, t2));


    // ====================================================
    printf("2) Running dyadic convolution using Fast Walsh Transform on device... ");
    cudaEventRecord(startStream1, stream1);
    
    // Full-precision
    fwtBatchGPU(d_Data, 1, log2Data, stream1);
    fwtBatchGPU(d_Kernel, 1, log2Data, stream1);
    modulateGPU(d_Data, d_Kernel, dataN, stream1);
    fwtBatchGPU(d_Data, 1, log2Data, stream1);
    // Reduced-precision
    fwtBatchGPU(d_Data_rp, 1, log2Data, stream1);
    fwtBatchGPU(d_Kernel_rp, 1, log2Data, stream1);
    modulateGPU(d_Data_rp, d_Kernel_rp, dataN, stream1);
    fwtBatchGPU(d_Data_rp, 1, log2Data, stream1);

    cudaEventRecord(stopStream1, stream1);
    cudaEventSynchronize(stopStream1);

    float msStream1 = 0;
    cudaEventElapsedTime(&msStream1, startStream1, stopStream1);
    printf("(%3.3lf ms) \n", msStream1);
    
    // ====================================================
    printf("    2.1) Reading back device results... ");
    getTimeNow(&t1);

    // Full-precision
    cudaMemcpyAsync(h_ResultGPU, d_Data, DATA_SIZE, cudaMemcpyDeviceToHost, stream1);
    // Reduced-precision
    cudaMemcpyAsync(h_ResultGPU_rp, d_Data_rp, DATA_SIZE_RP, cudaMemcpyDeviceToHost, stream1);

    cudaStreamSynchronize(stream1);

    getTimeNow(&t2);
    printf("(%3.3lf ms)\n", elapsedTime(t1, t2));

    // ====================================================
    printf("3) Running straightforard CPU dyadic convolution... ");
    getTimeNow(&t1);

    dyadicConvolutionCPU(h_ResultCPU, h_Data, h_Kernel, log2Data, log2Kernel);

    getTimeNow(&t2);
    printf("(%3.3lf ms)\n", elapsedTime(t1, t2));

    // ====================================================
    printf("4) Comparing the results... ");
    getTimeNow(&t1);

    sum_delta2 = 0;
    sum_ref2   = 0;
    for (i = 0; i < dataN; i++) {
        delta       = h_ResultCPU[i] - h_ResultGPU[i];
        ref         = h_ResultCPU[i];
        sum_delta2 += delta * delta;
        sum_ref2   += ref * ref;
    }
    L2norm = sqrt(sum_delta2 / sum_ref2);

    getTimeNow(&t2);
    printf("(%3.3lf ms)\n", elapsedTime(t1, t2));

    printf("    L2 norm: %E\n", L2norm);
    printf((L2norm < 1e-6) ? "    TEST PASSED\n" : "    TEST FAILED\n");

    // ====================================================
    printf("5) Comparing Double VS Float... ");
    getTimeNow(&t1);

    // Relative error
    relative_error_gpu(d_Data, d_Data_rp, d_Error, dataN);
    cudaMemcpy(h_Error, d_Error, DATA_SIZE_RP, cudaMemcpyDeviceToHost);    
    int iMaxRelErr = 0;
    for (i = 0; i < dataN; i++) if (h_Error[i] > h_Error[iMaxRelErr]) iMaxRelErr = i;
    // Absolute error
    for (i = 0; i < dataN; i++) h_Error[i] = abs(h_ResultGPU[i] - h_ResultGPU_rp[i]);
    int iMaxAbsErr = 0;
    for (i = 0; i < dataN; i++) if (h_Error[i] > h_Error[iMaxAbsErr]) iMaxAbsErr = i;
    

    getTimeNow(&t2);
    printf("(%3.3lf ms)\n", elapsedTime(t1, t2));

    printf("    Max relative error: %f (%f x %f)\n", h_Error[iMaxRelErr], h_ResultCPU[iMaxRelErr], h_ResultGPU_rp[iMaxRelErr]);
    printf("    Max absolute error: %f (%f x %f)\n", h_Error[iMaxAbsErr], h_ResultCPU[iMaxAbsErr], h_ResultGPU_rp[iMaxAbsErr]);

    // ====================================================
    printf("6) Shutting down\n");
    // Full-precision
    free(h_ResultGPU);
    free(h_ResultCPU);
    free(h_Data);
    free(h_Kernel);
    CHECK_CUDA_ERROR(cudaFree(d_Data));
    CHECK_CUDA_ERROR(cudaFree(d_Kernel));
    // Reduced-precision
    free(h_ResultGPU_rp);
    free(h_Data_rp);
    free(h_Kernel_rp);
    CHECK_CUDA_ERROR(cudaFree(d_Data_rp));
    CHECK_CUDA_ERROR(cudaFree(d_Kernel_rp));
    // Error calculation
    CHECK_CUDA_ERROR(cudaFree(d_Error));
}
