#include "fwtCPU.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

////////////////////////////////////////////////////////////////////////////////
// CPU FWT
////////////////////////////////////////////////////////////////////////////////
void fwtCPU(double *h_Output, double *h_Input, int log2N) {
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

void slowWTcpu(double *h_Output, double *h_Input, int log2N) {
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

bool validateGPUOutput(double *h_Data, double *h_Kernel, int log2Data, int log2Kernel, double *h_ResultGPU) {
    int dataN = 1 << log2Data;
    int i;
    
    double *h_ResultCPU = (double*)malloc(dataN * sizeof(double));

    dyadicConvolutionCPU(h_ResultCPU, h_Data, h_Kernel, log2Data, log2Kernel);

    double delta, ref, sum_delta2, sum_ref2, L2norm;
    sum_delta2 = 0;
    sum_ref2   = 0;
    for (i = 0; i < dataN; i++) {
        delta       = h_ResultCPU[i] - h_ResultGPU[i];
        ref         = h_ResultCPU[i];
        sum_delta2 += delta * delta;
        sum_ref2   += ref * ref;
    }
    L2norm = sqrt(sum_delta2 / sum_ref2);

    fprintf(stderr, "Validating output: ");
    fprintf(stderr, "TEST %s (L2 norm = %E)\n", ((L2norm < 1e-6) ? "PASSED" : "FAILED"), L2norm);

    free(h_ResultCPU);

    return L2norm < 1e-6;
}