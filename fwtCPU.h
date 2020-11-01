#ifndef FWT_CPU_H
#define FWT_CPU_H

void fwtCPU(double *h_Output, double *h_Input, int log2N);
void slowWTcpu(double *h_Output, double *h_Input, int log2N);
void dyadicConvolutionCPU(double *h_Result, double *h_Data, double *h_Kernel, int log2dataN, int log2kernelN);
bool validateGPUOutput(double *h_Data, double *h_Kernel, int log2Data, int log2Kernel, double *h_ResultGPU);

#endif