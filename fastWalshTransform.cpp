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

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#include <time.h>
#include <sys/time.h>

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

void slowWTcpu(double *h_Output, double *h_Input, int log2N){
    const int N = 1 << log2N;

    for(int i = 0; i < N; i++){
        double sum = 0;

        for(int j = 0; j < N; j++){
            //Walsh-Hadamar quotent
            double q = 1.0;
            for(int t = i & j; t != 0; t >>= 1)
                if(t & 1) q = -q;

            sum += q * h_Input[j];
        }

        h_Output[i] = (double)sum;
    }
}

void dyadicConvolutionCPU(double *h_Result, double *h_Data, double *h_Kernel, int log2dataN, int log2kernelN) {
    const int   dataN = 1 << log2dataN;
    const int kernelN = 1 << log2kernelN;

    for(int i = 0; i < dataN; i++){
        double sum = 0;

        for(int j = 0; j < kernelN; j++)
            sum += h_Data[i ^ j] * h_Kernel[j];

        h_Result[i] = (double)sum;
    }
}


////////////////////////////////////////////////////////////////////////////////
// GPU FWT
////////////////////////////////////////////////////////////////////////////////
extern void fwtBatchGPU(double *d_Data, int M, int log2N);
extern void fwtBatchGPU(float *d_Data, int M, int log2N);
extern void modulateGPU(double *d_A, double *d_B, int N);
extern void modulateGPU(float *d_A, float *d_B, int N);

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
    double *h_Data, *h_Kernel, *h_ResultCPU, *h_ResultGPU;
    float *h_Data_rp, *h_Kernel_rp, *h_ResultGPU_rp;
    float *h_Error;

    double *d_Data, *d_Kernel;
    float *d_Data_rp, *d_Kernel_rp;
    float *d_Error;

    double delta, ref, sum_delta2, sum_ref2, L2norm;
    int i;

    Time t1, t2, t3;

    printf("1) Initializing data...\n");
    printf("    1.1) Allocating CPU memory\n");
    h_Kernel    = (double*)malloc(KERNEL_SIZE);
    h_Data      = (double*)malloc(DATA_SIZE);
    h_ResultCPU = (double*)malloc(DATA_SIZE);
    h_ResultGPU = (double*)malloc(DATA_SIZE);

    h_Kernel_rp    = (float*)malloc(KERNEL_SIZE_RP);
    h_Data_rp      = (float*)malloc(DATA_SIZE_RP);
    h_ResultGPU_rp = (float*)malloc(DATA_SIZE_RP);

    h_Error = (float*)malloc(DATA_SIZE_RP);

    printf("    1.2) Allocating GPU memory\n");
    cudaMalloc((void**)&d_Kernel, DATA_SIZE);
    cudaMalloc((void**)&d_Data,   DATA_SIZE);

    cudaMalloc((void**)&d_Kernel_rp, DATA_SIZE_RP);
    cudaMalloc((void**)&d_Data_rp,   DATA_SIZE_RP);

    cudaMalloc((void**)&d_Error,   DATA_SIZE_RP);

    printf("    1.3) Generating data\n");
    srand((int)time(NULL));
    for (i = 0; i < kernelN; i++) {
        h_Kernel[i] = (double)rand() / (double)RAND_MAX;
        h_Kernel_rp[i] = h_Kernel[i];
    }

    for (i = 0; i < dataN; i++) {
        h_Data[i] = (double)rand() / (double)RAND_MAX;
        h_Data_rp[i] = h_Data[i];
    }

    cudaMemset(d_Kernel, 0, DATA_SIZE);
    cudaMemcpy(d_Kernel, h_Kernel, KERNEL_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Data, h_Data, DATA_SIZE, cudaMemcpyHostToDevice);

    cudaMemset(d_Kernel_rp, 0, DATA_SIZE_RP);
    cudaMemcpy(d_Kernel_rp, h_Kernel_rp, KERNEL_SIZE_RP, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Data_rp, h_Data_rp, DATA_SIZE_RP, cudaMemcpyHostToDevice);


    printf("2) Running GPU dyadic convolution using Fast Walsh Transform\n");
    cudaThreadSynchronize();
    // Running FP64
    getTimeNow(&t1);

    fwtBatchGPU(d_Data, 1, log2Data);
    fwtBatchGPU(d_Kernel, 1, log2Data);
    modulateGPU(d_Data, d_Kernel, dataN);
    fwtBatchGPU(d_Data, 1, log2Data);
    //cudaThreadSynchronize();

    //  getTimeNow(&t2);
    //  printf("    FP64 execution: %lf ms\n", elapsedTime(t1, t2));

    // Running FP32
    fwtBatchGPU(d_Data_rp, 1, log2Data);
    fwtBatchGPU(d_Kernel_rp, 1, log2Data);
    modulateGPU(d_Data_rp, d_Kernel_rp, dataN);
    fwtBatchGPU(d_Data_rp, 1, log2Data);
    cudaThreadSynchronize();

    getTimeNow(&t3);
    printf("    FP64 + FP32 execution: %lf ms\n", elapsedTime(t1, t3));

    printf("    2.1) Reading back GPU results\n");
    cudaMemcpy(h_ResultGPU, d_Data, DATA_SIZE, cudaMemcpyDeviceToHost);

    cudaMemcpy(h_ResultGPU_rp, d_Data_rp, DATA_SIZE_RP, cudaMemcpyDeviceToHost);


    printf("3) Running straightforward CPU dyadic convolution\n");
    dyadicConvolutionCPU(h_ResultCPU, h_Data, h_Kernel, log2Data, log2Kernel);


    printf("4) Comparing the results\n");
    sum_delta2 = 0;
    sum_ref2   = 0;
    for (i = 0; i < dataN; i++) {
        delta       = h_ResultCPU[i] - h_ResultGPU[i];
        ref         = h_ResultCPU[i];
        sum_delta2 += delta * delta;
        sum_ref2   += ref * ref;
    }
    L2norm = sqrt(sum_delta2 / sum_ref2);
    printf("    L2 norm: %E\n", L2norm);
    printf((L2norm < 1e-6) ? "    TEST PASSED\n" : "    TEST FAILED\n");


    printf("5) Comparing Double VS Float\n");
    // Relative error
    relative_error_gpu(d_Data, d_Data_rp, d_Error, dataN);
    cudaMemcpy(h_Error, d_Error, DATA_SIZE_RP, cudaMemcpyDeviceToHost);    
    int iMaxRelErr = 0;
    for (i = 0; i < dataN; i++) if (h_Error[i] > h_Error[iMaxRelErr]) iMaxRelErr = i;
    printf("    Max relative error: %f (%f x %f)\n", h_Error[iMaxRelErr], h_ResultCPU[iMaxRelErr], h_ResultGPU_rp[iMaxRelErr]);
    // Absolute error
    for (i = 0; i < dataN; i++) h_Error[i] = abs(h_ResultGPU[i] - h_ResultGPU_rp[i]);
    int iMaxAbsErr = 0;
    for (i = 0; i < dataN; i++) if (h_Error[i] > h_Error[iMaxAbsErr]) iMaxAbsErr = i;
    printf("    Max absolute error: %f (%f x %f)\n", h_Error[iMaxAbsErr], h_ResultCPU[iMaxAbsErr], h_ResultGPU_rp[iMaxAbsErr]);

    printf("6) Shutting down\n");
    cudaFree(d_Data);
    cudaFree(d_Kernel);
    free(h_ResultGPU);
    free(h_ResultCPU);
    free(h_Data);
    free(h_Kernel);

    cudaFree(d_Data_rp);
    cudaFree(d_Kernel_rp);
    free(h_ResultGPU_rp);
    free(h_Data_rp);
    free(h_Kernel_rp);
}
