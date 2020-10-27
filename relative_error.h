#ifndef RELATIVE_ERROR_H
#define REALTIVE_ERROR_H

#define MIN_PERCENTAGE 0.99f
#define MAX_PERCENTAGE 1.01f

void check_relative_error_gpu(double *array, float *array_rp, int N);
void calc_relative_error_gpu(double *array, float *array_rp, float *err_out, int N);
unsigned long long get_dmr_error();

#endif