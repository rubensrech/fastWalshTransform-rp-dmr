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
#include <string>
#include <string.h>
#include <regex>
#include <bitset>
#include <iostream>

#include "util.h"
#include "fwtCPU.h"

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
const int KERNEL_SIZE = kernelN * sizeof(double);

#if DMR_TYPE != NO_DMR
    const int DATA_SIZE_RP = dataN   * sizeof(dmr_t);
    const int KERNEL_SIZE_RP = kernelN * sizeof(dmr_t);
#endif

////////////////////////////////////////////////////////////////////////////////
// GPU FWT
////////////////////////////////////////////////////////////////////////////////
#include "calc_error.h"

extern void fwtBatchGPU(double *d_Data, int M, int log2N, cudaStream_t stream);
extern void modulateGPU(double *d_A, double *d_B, int N, cudaStream_t stream);

#if DMR_TYPE != NO_DMR
    extern void fwtBatchGPU(dmr_t *d_Data, int M, int log2N, cudaStream_t stream);
    extern void modulateGPU(dmr_t *d_A, dmr_t *d_B, int N, cudaStream_t stream);
    extern void duplicate_input_gpu(double *input, dmr_t *input_rp, int N);
#endif

////////////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char *argv[]) {
    // ====================================================
    // > Managing arguments
    // * Load Input
    char *input_filename = find_char_arg(argc, argv, (char*)"-input", (char*)"inputs/input-bit-21.data");
    bool loadInput = true;
    // * Measure time
    bool measureTime = find_int_arg(argc, argv, (char*)"-measureTime", 0);
    // * Iterations
    int iterations = find_int_arg(argc, argv, (char*)"-it", 50);

#if DMR_TYPE == NO_DMR
    printf("> DMR: NO-DMR\n");
#elif DMR_TYPE == DMR_FLOAT
    printf("> DMR: Float\n");
#elif DMR_TYPE == DMR_DOUBLE
    printf("> DMR: Double\n");
#endif

    // ====================================================
    // > Declaring variables
    double *h_Data, *h_Kernel, *h_ResultGPU;
    double *d_Data, *d_Kernel;

#if DMR_TYPE != NO_DMR
    dmr_t *h_Data_rp, *h_Kernel_rp, *h_ResultGPU_rp;
    dmr_t *d_Data_rp, *d_Kernel_rp;
#endif

    //      - Extra
    cudaStream_t stream1;
    cudaEvent_t start, stop;

    Time t0, t1;
    int i;

    if (measureTime) getTimeNow(&t0);

    // ====================================================
    // > Allocating CPU memory
    h_Kernel    = (double*)malloc(KERNEL_SIZE);
    h_Data      = (double*)malloc(DATA_SIZE);
    h_ResultGPU = (double*)malloc(DATA_SIZE);

#if DMR_TYPE != NO_DMR
    h_Kernel_rp    = (dmr_t*)malloc(KERNEL_SIZE_RP);
    h_Data_rp      = (dmr_t*)malloc(DATA_SIZE_RP);
    h_ResultGPU_rp = (dmr_t*)malloc(DATA_SIZE_RP);
#endif

    // ====================================================
    // > Allocating GPU memory

    CHECK_CUDA_ERROR(cudaStreamCreateWithFlags(&stream1, cudaStreamNonBlocking));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_Kernel, DATA_SIZE));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_Data,   DATA_SIZE));

#if DMR_TYPE != NO_DMR
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_Kernel_rp, DATA_SIZE_RP));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_Data_rp,   DATA_SIZE_RP));
#endif

    // ====================================================
    // > Generating/Loading input data

    if (loadInput) {
        // > Loading input data
        load_input(input_filename, h_Data, dataN, h_Kernel, kernelN);
    } else {
        // > Generating input data
        srand((int)time(NULL));
        for (i = 0; i < kernelN; i++) h_Kernel[i] = (double)rand() / (double)RAND_MAX;
        for (i = 0; i < dataN; i++) h_Data[i] = (double)rand() / (double)RAND_MAX;
    }

    // ====================================================
    // > Copying data to device
    float memCpyToDeviceTimeMs;
    if (measureTime) {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);
    }

    // Full-precision    
    CHECK_CUDA_ERROR(cudaMemsetAsync(d_Kernel, 0, DATA_SIZE, stream1));
    CHECK_CUDA_ERROR(cudaMemcpyAsync(d_Kernel, h_Kernel, KERNEL_SIZE, cudaMemcpyHostToDevice, stream1));
    CHECK_CUDA_ERROR(cudaMemcpyAsync(d_Data, h_Data, DATA_SIZE, cudaMemcpyHostToDevice, stream1));
    cudaStreamSynchronize(stream1);

    if (measureTime) {
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&memCpyToDeviceTimeMs, start, stop);
        printf("%s* MemCpy to device: %f ms %s\n", GREEN, memCpyToDeviceTimeMs, DFT_COLOR);
    }

#if DMR_TYPE != NO_DMR
    float dupInputTimeMs;
    if (measureTime) {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);
    }

    // ====================================================
    // > Duplicating input
    duplicate_input_gpu(d_Kernel, d_Kernel_rp, kernelN);
    duplicate_input_gpu(d_Data, d_Data_rp, dataN);

    if (measureTime) {
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&dupInputTimeMs, start, stop);
        printf("%s* Input duplication: %f ms %s\n", GREEN, dupInputTimeMs, DFT_COLOR);
    }
#endif

    // ====================================================
    // > Running Fast Walsh Transform on device

    float kernelsTimeMs;
    if (measureTime) {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);
    }
    
    for (i = 0; i < iterations; i++) {
        fwtBatchGPU(d_Data, 1, log2Data, stream1);
        fwtBatchGPU(d_Kernel, 1, log2Data, stream1);
        modulateGPU(d_Data, d_Kernel, dataN, stream1);
        fwtBatchGPU(d_Data, 1, log2Data, stream1);

#if DMR_TYPE != NO_DMR
        fwtBatchGPU(d_Data_rp, 1, log2Data, stream1);
        fwtBatchGPU(d_Kernel_rp, 1, log2Data, stream1);
        modulateGPU(d_Data_rp, d_Kernel_rp, dataN, stream1);
        fwtBatchGPU(d_Data_rp, 1, log2Data, stream1);  
#endif  
    }
    
    if (measureTime) {
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&kernelsTimeMs, start, stop);
        printf("%s* Kernels time: %f ms %s\n", GREEN, kernelsTimeMs, DFT_COLOR);
    }


#if DMR_TYPE != NO_DMR
    // ====================================================
    // > Checking for faults

    float checkFaultsTimeMs;
    if (measureTime) {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);
    }

    check_errors_gpu(d_Data, d_Data_rp, dataN);

    if (measureTime) {
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&checkFaultsTimeMs, start, stop);
        printf("%s* Check faults: %f ms %s\n", GREEN, checkFaultsTimeMs, DFT_COLOR);
    }
#endif  

    // ====================================================
    // > Reading back device results

    float memCpyToHostTimeMs;
    if (measureTime) {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);
    }

    cudaMemcpyAsync(h_ResultGPU, d_Data, DATA_SIZE, cudaMemcpyDeviceToHost, stream1);
    cudaStreamSynchronize(stream1);

    if (measureTime) {
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&memCpyToHostTimeMs, start, stop);
        printf("%s* MemCpy to host: %f ms %s\n", GREEN, memCpyToHostTimeMs, DFT_COLOR);
    }

#if DMR_TYPE != NO_DMR
    unsigned long long dmrErrors = get_dmr_error();
    bool faultDetected = dmrErrors > 0;
    // printf("> Faults detected?  %s (DMR errors: %llu)\n", faultDetected ? "YES" : "NO", dmrErrors);
#endif

    // ====================================================
    // > Shutting down
    free(h_ResultGPU);
    free(h_Data);
    free(h_Kernel);
    CHECK_CUDA_ERROR(cudaFree(d_Data));
    CHECK_CUDA_ERROR(cudaFree(d_Kernel));

#if DMR_TYPE != NO_DMR
    // Reduced-precision
    free(h_ResultGPU_rp);
    free(h_Data_rp);
    free(h_Kernel_rp);
    CHECK_CUDA_ERROR(cudaFree(d_Data_rp));
    CHECK_CUDA_ERROR(cudaFree(d_Kernel_rp));
#endif

    if (measureTime) {
        getTimeNow(&t1);
        printf("%s* Total execution time: %.3lf ms (%d iterations)%s\n", GREEN, elapsedTime(t0, t1), iterations, DFT_COLOR);
    }

    return 0;
}
