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
const int DATA_SIZE_RP = dataN   * sizeof(float);
const int KERNEL_SIZE = kernelN * sizeof(double);
const int KERNEL_SIZE_RP = kernelN * sizeof(float);

////////////////////////////////////////////////////////////////////////////////
// GPU FWT
////////////////////////////////////////////////////////////////////////////////
#include "calc_error.h"

extern void fwtBatchGPU(double *d_Data, int M, int log2N, cudaStream_t stream);
extern void fwtBatchGPU(float *d_Data, int M, int log2N, cudaStream_t stream);
extern void modulateGPU(double *d_A, double *d_B, int N, cudaStream_t stream);
extern void modulateGPU(float *d_A, float *d_B, int N, cudaStream_t stream);

extern void relative_error_gpu(double *output, float *output_rp, float *err_output, int N);

////////////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char *argv[]) {
    // ====================================================
    // > Managing arguments
    // * Load Input
    char *input_filename = find_char_arg(argc, argv, (char*)"-input", (char*)"none");
    bool loadInput = (strcmp(input_filename, (char*)"none")==0) ? false : true;
    // * Save input
    bool saveInput = find_int_arg(argc, argv, (char*)"-saveInput", 0);
    // * Save input UINT thresh
    int saveInputBitThresh = find_int_arg(argc, argv, (char*)"-saveInputBitThresh", 0);
    // * Save output
    bool saveOutput = find_int_arg(argc, argv, (char*)"-saveOutput", 0);
    // * Validate output
    bool validateOutput = find_int_arg(argc, argv, (char*)"-validateOutput", 0);
    // * Measure time
    bool measureTime = find_int_arg(argc, argv, (char*)"-measureTime", 0);

    // ====================================================
    // > Declaring variables
    // * Host data
    //      - Full-precision
    double *h_Data, *h_Kernel, *h_ResultGPU;
    //      - Reduced-precision
    float *h_Data_rp, *h_Kernel_rp, *h_ResultGPU_rp;
    //      - Extra
    float *h_Error;

    // * Device data
    //      - Full-precision
    double *d_Data, *d_Kernel;
    //      - Reduced-precision
    float *d_Data_rp, *d_Kernel_rp;
    //      - Extra
    float *d_Error;
    cudaStream_t stream1;
    cudaEvent_t startStream1, stopStream1;

    // * CPU version
    double *h_ResultCPU;
    double delta, ref, sum_delta2, sum_ref2, L2norm;
    Time t0;
    int i;

    if (measureTime) getTimeNow(&t0);

    // ====================================================
    // > Allocating CPU memory

    cudaEventCreate(&startStream1);
    cudaEventCreate(&stopStream1);

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

    // ====================================================
    // > Allocating GPU memory

    CHECK_CUDA_ERROR(cudaStreamCreateWithFlags(&stream1, cudaStreamNonBlocking));
    // Full-precision
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_Kernel, DATA_SIZE));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_Data,   DATA_SIZE));
    // Reduced-precision
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_Kernel_rp, DATA_SIZE_RP));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_Data_rp,   DATA_SIZE_RP));
    // Error calculation
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_Error,   DATA_SIZE_RP));

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
    // > Duplicating input
    for (i = 0; i < kernelN; i++)   h_Kernel_rp[i] = h_Kernel[i];
    for (i = 0; i < dataN; i++)     h_Data_rp[i] = h_Data[i];

    // ====================================================
    // > Copying data to device

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

    // ====================================================
    // > Running Fast Walsh Transform on device

    // cudaEventRecord(startStream1, stream1);
    
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

    // float msStream1 = 0;
    // cudaEventElapsedTime(&msStream1, startStream1, stopStream1);
    // printf("Kernels elapsed time: %3.3lf ms\n", msStream1);
    
    // ====================================================
    // > Reading back device results

    // Full-precision
    cudaMemcpyAsync(h_ResultGPU, d_Data, DATA_SIZE, cudaMemcpyDeviceToHost, stream1);
    // Reduced-precision
    cudaMemcpyAsync(h_ResultGPU_rp, d_Data_rp, DATA_SIZE_RP, cudaMemcpyDeviceToHost, stream1);

    cudaStreamSynchronize(stream1);

    // ====================================================
    // > Validating output
    if (validateOutput) {
        validateGPUOutput(h_Data, h_Kernel, log2Data, log2Kernel, h_ResultGPU);
    }

    // ====================================================
    // > Saving input

    // unsigned int maxUINTError = std::max(get_max_uint_error_non_zeros(), get_max_uint_error_zeros());
    // int maxErrorBit = log2_host(maxUINTError);
    // unsigned int UINTThresh = 1 << saveInputBitThresh;
    // bool inputSaved = false;
    // if (saveInput && maxUINTError <= UINTThresh) {
    //     if (!save_input(h_Data, dataN, h_Kernel, kernelN, maxErrorBit)) {
    //         fprintf(stderr, "ERROR: could not save input\n");
    //     } else {
    //         inputSaved = true;
    //     }
    // }
    bool inputSaved = false;
    int maxErrorBit = -1;

    // ====================================================
    // > Saving output
    if (saveOutput) {
        if (save_output(h_ResultGPU, dataN)) {
            printf("OUTPUT SAVED SUCCESSFULY\n");
        } else {
            fprintf(stderr, "ERROR: could not save output\n");
        }
    }

#ifdef FIND_THRESHOLD
    // ====================================================
    // > Finding thresholds

    float *relErrArr = (float*)calloc(sizeof(float), dataN);
    float *absErrArr = (float*)calloc(sizeof(float), dataN);

    calc_errors_gpu(d_Data, d_Data_rp, dataN);

    get_rel_error_array(relErrArr, dataN);
    get_abs_error_array(absErrArr, dataN);

    int ignoreRelErrCount = 0;
    int iMaxRelErr = find_max_i(relErrArr, dataN);
    int iMaxAbsErr = find_max_i(absErrArr, dataN);

    for (i = 0; i < dataN; i++) if (relErrArr[i] == IGNORE_VAL_FLAG) ignoreRelErrCount++;

    printf("> MAX ERRORS\n");
    std::cout << "    UINT error:  " << std::bitset<32>(get_max_uint_err()) << std::endl;
    printf("    RELATIVE error:  %f (%e X %e) - ignored vals: %d\n", relErrArr[iMaxRelErr], h_ResultGPU[iMaxRelErr], h_ResultGPU_rp[iMaxRelErr], ignoreRelErrCount);
    printf("    ABSOLUTE error:  %f (%e X %e)\n", absErrArr[iMaxAbsErr], h_ResultGPU[iMaxAbsErr], h_ResultGPU_rp[iMaxAbsErr]);

    printf("> GENERAL STATISTICS\n");
    printf("    Zeros FP64: %u\n", get_zeros_fp64());
    printf("    Zeros FP32: %u\n", get_zeros_fp32());
    printf("    Negatives: %u\n", get_negatives());

#else
    // ====================================================
    // > Checking for faults

    check_errors_gpu(d_Data, d_Data_rp, dataN);

    printf("> Error metric: %s\n", ERROR_METRIC == HYBRID ? "Hybrid (Rel + Abs)" : (ERROR_METRIC == UINT_ERROR ? "UINT Error" : "Relative Error"));

    unsigned long long dmrErrors = get_dmr_error();
    bool faultDetected = dmrErrors > 0;
    printf("> Faults detected?  %s (DMR errors: %llu)\n", faultDetected ? "YES" : "NO", dmrErrors);

    // ====================================================
    // > Checking output against Golden output
    std::string gold_output_filename(input_filename);
    gold_output_filename = std::regex_replace(gold_output_filename, std::regex("input"), "output");
    bool outputIsCorrect = compare_output_with_golden(h_ResultGPU, dataN, gold_output_filename.c_str());
    printf("> Output corrupted? %s\n", !outputIsCorrect ? "YES" : "NO");

    // ====================================================
    // > Classifing
    printf("> DMR classification: ");
    if (faultDetected && outputIsCorrect) printf("FALSE POSITIVE\n");
    if (faultDetected && !outputIsCorrect) printf("TRUE POSITIVE\n");
    if (!faultDetected && outputIsCorrect) printf("TRUE NEGATIVE\n");
    if (!faultDetected && !outputIsCorrect) printf("FALSE NEGATIVE\n");
    
#endif

    if (inputSaved) {
        printf("\nINPUT SAVED SUCCESSFULLY! (Max error bit: %d)\n", maxErrorBit);
        exit(5);
    }

    // ====================================================
    // > Shutting down
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
