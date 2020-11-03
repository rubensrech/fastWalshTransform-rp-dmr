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
// GPU FUNCTIONS
////////////////////////////////////////////////////////////////////////////////
extern void fwtBatchGPU(double *d_Data, float *d_Output_rp, int M, int log2N, float *h_err, float *d_err, std::vector<float> &max_errs);
extern void modulateGPU(double *d_A, float *d_A_rp, double *d_B, int N, float *h_err, float *d_err, std::vector<float> &max_errs);

extern void check_relative_error_gpu(double *array, float *array_rp, int N);
extern void calc_relative_error_gpu(double *array, float *array_rp, float *err_out, int N);
extern unsigned long long get_dmr_error();

extern void copyGPU(float *array_rp, double *array, int N);



////////////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char *argv[]) {
    // ====================================================
    // > Managing arguments
    // * Load Input
    char *input_filename = find_char_arg(argc, argv, (char*)"-input", (char*)"none");
    bool loadInput = (strcmp(input_filename, (char*)"none")==0) ? false : true;
    // * Save output
    bool saveOutput = find_int_arg(argc, argv, (char*)"-saveOutput", 0);
    // * Validate output
    bool validateOutput = find_int_arg(argc, argv, (char*)"-validateOutput", 0);
    // * Measure time
    bool measureTime = find_int_arg(argc, argv, (char*)"-measureTime", 0);

    // ====================================================
    // > Declaring variables

    // * Host data
    // Full-precision
    double *h_Data, *h_Kernel, *h_ResultGPU;
    // Reduced-precision
    float *h_ResultGPU_rp;
    // Error calculation
    float *h_Error;
    std::vector<float>h_maxRelErrs;

    // * Device data
    // Full-precision
    double *d_Data, *d_Kernel;
    // Reduced-precision
    float *d_Data_rp, *d_Kernel_rp;
    // Error calculation
    float *d_Error;

    Time t0, t1;
    int i;

    if (measureTime) getTimeNow(&t0);

    // ====================================================
    // > Allocating CPU memory

    // Full-precision
    h_Kernel    = (double*)malloc(KERNEL_SIZE);
    h_Data      = (double*)malloc(DATA_SIZE);
    h_ResultGPU = (double*)malloc(DATA_SIZE);
    // Reduced-precision
    h_ResultGPU_rp = (float*)malloc(DATA_SIZE_RP);
    // Error calculation
    h_Error = (float*)malloc(DATA_SIZE_RP);
    

    // ====================================================
    // > Allocating GPU memory

    // Full-precision
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_Kernel,     DATA_SIZE));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_Data,       DATA_SIZE));
    // Reduced-precision
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_Data_rp,    DATA_SIZE_RP));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_Kernel_rp,  DATA_SIZE_RP));
    // Error calculation
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_Error, DATA_SIZE_RP));

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

    // Full-precision    
    CHECK_CUDA_ERROR(cudaMemsetAsync(d_Kernel, 0, DATA_SIZE));
    CHECK_CUDA_ERROR(cudaMemcpyAsync(d_Kernel, h_Kernel, KERNEL_SIZE, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpyAsync(d_Data, h_Data, DATA_SIZE, cudaMemcpyHostToDevice));
    
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    // ====================================================
    // > Running Fast Walsh Transform on device

    // Full-precision
    fwtBatchGPU(d_Data, d_Data_rp, 1, log2Data, h_Error, d_Error, h_maxRelErrs);
    fwtBatchGPU(d_Kernel, d_Kernel_rp, 1, log2Data, h_Error, d_Error, h_maxRelErrs);
    modulateGPU(d_Data, d_Data_rp, d_Kernel, dataN, h_Error, d_Error, h_maxRelErrs);
    fwtBatchGPU(d_Data, d_Data_rp, 1, log2Data, h_Error, d_Error, h_maxRelErrs);

    // ====================================================
    // > Reading back device results

    // Full-precision
    cudaMemcpy(h_ResultGPU, d_Data, DATA_SIZE, cudaMemcpyDeviceToHost);
    // Reduced-precision
    cudaMemcpy(h_ResultGPU_rp, d_Data_rp, DATA_SIZE_RP, cudaMemcpyDeviceToHost);

    // ====================================================
    // > Validating output
    if (validateOutput) {
        validateGPUOutput(h_Data, h_Kernel, log2Data, log2Kernel, h_ResultGPU);
    }
    
    // ====================================================
    // > Saving output
    if (saveOutput) {
        if (!save_output(h_ResultGPU, dataN)) {
            fprintf(stderr, "ERROR: could not save output\n");
        }
    }

    // ====================================================
    // > Checking for faults
    unsigned long long dmrErrors = get_dmr_error();
    bool faultDetected = dmrErrors > 0;
    printf("> DMR errors: %llu\n", dmrErrors);

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

    // ====================================================
    // > Shutting down

    // Full-precision
    free(h_ResultGPU);
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

    if (measureTime) {
        getTimeNow(&t1);
        printf("> Total execution time: %.3lf ms\n", elapsedTime(t0, t1));
    }
}
