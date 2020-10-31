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
extern void fwtBatchGPU(double *d_Data, float *d_Output_rp, int M, int log2N, float *h_err, float *d_err, std::vector<float> &max_errs);
extern void modulateGPU(double *d_A, float *d_A_rp, double *d_B, int N, float *h_err, float *d_err, std::vector<float> &max_errs);

extern void check_relative_error_gpu(double *array, float *array_rp, int N);
extern void calc_relative_error_gpu(double *array, float *array_rp, float *err_out, int N);
extern unsigned long long get_dmr_error();

extern void copyGPU(float *array_rp, double *array, int N);

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

    // ===== Arguments =====
    // > Load Input
    char *input_filename = find_char_arg(argc, argv, (char*)"-input", (char*)"none");
    bool loadInput = (strcmp(input_filename, (char*)"none")==0) ? false : true;

    // > Save output
    bool saveOutput = find_int_arg(argc, argv, (char*)"-saveOutput", 0);

    // Host data
    // Full-precision
    double *h_Data, *h_Kernel, *h_ResultCPU, *h_ResultGPU;
    // Reduced-precision
    float *h_ResultGPU_rp;

    float *h_Error;
    std::vector<float>h_maxRelErrs;

    // Device data
    // Full-precision
    double *d_Data, *d_Kernel;
    // Reduced-precision
    float *d_Data_rp, *d_Kernel_rp;

    float *d_Error;
    cudaEvent_t startEvent, stopEvent;

    double delta, ref, sum_delta2, sum_ref2, L2norm;
    Time t1, t2;
    int i;

    // ==========================================================================
    // ==========================================================================
    printf("1) Initializing data\n");

    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);

    // ====================================================
    printf("    1.1) Allocating CPU memory... ");
    getTimeNow(&t1);

    // Full-precision
    h_Kernel    = (double*)malloc(KERNEL_SIZE);
    h_Data      = (double*)malloc(DATA_SIZE);
    h_ResultCPU = (double*)malloc(DATA_SIZE);
    h_ResultGPU = (double*)malloc(DATA_SIZE);
    // Reduced-precision
    h_ResultGPU_rp = (float*)malloc(DATA_SIZE_RP);
    // Error calculation
    h_Error = (float*)malloc(DATA_SIZE_RP);

    getTimeNow(&t2);
    printf("(%3.3lf ms)\n", elapsedTime(t1, t2));
    

    // ====================================================
    printf("    1.2) Allocating GPU memory... ");
    getTimeNow(&t1);

    // Full-precision
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_Kernel,     DATA_SIZE));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_Data,       DATA_SIZE));
    // Reduced-precision
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_Data_rp,    DATA_SIZE_RP));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_Kernel_rp,  DATA_SIZE_RP));
    // Error calculation
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_Error, DATA_SIZE_RP));

    getTimeNow(&t2);
    printf("(%3.3lf ms)\n", elapsedTime(t1, t2));

    // ====================================================
    if (loadInput) {
        printf("    1.3) Loading input data...");
        getTimeNow(&t1);

        int dN = 0, kN = 0;
        load_input(input_filename, h_Data, &dN, h_Kernel, &kN);

        getTimeNow(&t2);
        printf("(%3.3lf ms)\n", elapsedTime(t1, t2));

        if (dN != dataN || kN != kernelN) {
            printf("ERROR: Input data doesn't match the expected size\n");
            exit(-1);
        }
    } else {
        printf("    1.3) Generating data... ");
        getTimeNow(&t1);

        srand((int)time(NULL));
        for (i = 0; i < kernelN; i++) {
            h_Kernel[i] = (double)rand() / (double)RAND_MAX;
        }
        for (i = 0; i < dataN; i++) {
            h_Data[i] = (double)rand() / (double)RAND_MAX;
        }

        getTimeNow(&t2);
        printf("(%3.3lf ms)\n", elapsedTime(t1, t2));
    }
    // ====================================================
    printf("    1.4) Copying data to device... ");
    getTimeNow(&t1);

    // Full-precision    
    CHECK_CUDA_ERROR(cudaMemsetAsync(d_Kernel, 0, DATA_SIZE));
    CHECK_CUDA_ERROR(cudaMemcpyAsync(d_Kernel, h_Kernel, KERNEL_SIZE, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpyAsync(d_Data, h_Data, DATA_SIZE, cudaMemcpyHostToDevice));
    
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    getTimeNow(&t2);
    printf("(%3.3lf ms)\n", elapsedTime(t1, t2));


    // ====================================================
    printf("2) Running dyadic convolution using Fast Walsh Transform on device... ");
    cudaEventRecord(startEvent);

    // Full-precision
    fwtBatchGPU(d_Data, d_Data_rp, 1, log2Data, h_Error, d_Error, h_maxRelErrs);
    fwtBatchGPU(d_Kernel, d_Kernel_rp, 1, log2Data, h_Error, d_Error, h_maxRelErrs);
    modulateGPU(d_Data, d_Data_rp, d_Kernel, dataN, h_Error, d_Error, h_maxRelErrs);
    fwtBatchGPU(d_Data, d_Data_rp, 1, log2Data, h_Error, d_Error, h_maxRelErrs);

    cudaEventRecord(stopEvent);
    cudaEventSynchronize(stopEvent);

    float ms = 0;
    cudaEventElapsedTime(&ms, startEvent, stopEvent);
    printf("(%3.3lf ms) \n", ms);
    
    // ====================================================
    printf("    2.1) Reading back device results... ");
    getTimeNow(&t1);

    // Full-precision
    cudaMemcpyAsync(h_ResultGPU, d_Data, DATA_SIZE, cudaMemcpyDeviceToHost);
    // Reduced-precision
    cudaMemcpyAsync(h_ResultGPU_rp, d_Data_rp, DATA_SIZE_RP, cudaMemcpyDeviceToHost);

    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    getTimeNow(&t2);
    printf("(%3.3lf ms)\n", elapsedTime(t1, t2));

    // ====================================================
    // printf("3) Running straightforard CPU dyadic convolution... ");
    // getTimeNow(&t1);

    // dyadicConvolutionCPU(h_ResultCPU, h_Data, h_Kernel, log2Data, log2Kernel);

    // getTimeNow(&t2);
    // printf("(%3.3lf ms)\n", elapsedTime(t1, t2));

    // ====================================================
    printf("4) Comparing the results... ");
    getTimeNow(&t1);

    sum_delta2 = 0;
    sum_ref2   = 0;
    for (i = 0; i < dataN; i++) {
        delta       = h_ResultCPU[i] - h_ResultGPU_rp[i];
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
    calc_relative_error_gpu(d_Data, d_Data_rp, d_Error, dataN);
    cudaMemcpy(h_Error, d_Error, DATA_SIZE_RP, cudaMemcpyDeviceToHost);    
    int iMaxRelErr = 0;
    float maxRelErr = h_Error[0];
    for (i = 0; i < dataN; i++) if (h_Error[i] > maxRelErr) { iMaxRelErr = i; maxRelErr = h_Error[i]; }
    // Absolute error
    for (i = 0; i < dataN; i++) h_Error[i] = abs(h_ResultGPU[i] - h_ResultGPU_rp[i]);
    float maxAbsErr = h_Error[0];
    int iMaxAbsErr = 0;
    for (i = 0; i < dataN; i++) if (h_Error[i] > maxAbsErr) { iMaxAbsErr = i; maxAbsErr = h_Error[i]; }
    
    getTimeNow(&t2);
    printf("(%3.3lf ms)\n", elapsedTime(t1, t2));

    float maxErr = find_max(&h_maxRelErrs[0], h_maxRelErrs.size());
    printf("    Max relative errors (%d iterations): %1.2f", int(h_maxRelErrs.size()), h_maxRelErrs[0]);
    for (i = 1; i < h_maxRelErrs.size(); i++) printf(", %1.2f", h_maxRelErrs[i]);
    printf(" (max: %1.3f)\n", maxErr);
    
    printf("    Output max relative error: %f (%f x %f)\n", maxRelErr, h_ResultCPU[iMaxRelErr], h_ResultGPU_rp[iMaxRelErr]);
    printf("    Output max absolute error: %f (%f x %f)\n", maxAbsErr, h_ResultCPU[iMaxAbsErr], h_ResultGPU_rp[iMaxAbsErr]);
    printf("    DMR errors: %llu\n", get_dmr_error());


    // ====================================================
    if (maxErr < 1.25 && !loadInput) {
        printf("\nSaving input... ");
        bool inputSaved = save_input(h_Data, dataN, h_Kernel, kernelN, maxErr);
        printf(inputSaved ? "SAVED" : "FAILED");
        printf("\n\n");
    }

    if (saveOutput) {
        printf("\nSaving output... ");
        bool outputSaved = save_output(h_ResultGPU, dataN, maxErr);
        printf(outputSaved ? "SAVED" : "FAILED");
        printf("\n\n");
    }

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
    CHECK_CUDA_ERROR(cudaFree(d_Data_rp));
    CHECK_CUDA_ERROR(cudaFree(d_Kernel_rp));
    // Error calculation
    CHECK_CUDA_ERROR(cudaFree(d_Error));
}
