#ifndef RELATIVE_ERROR_H
#define REALTIVE_ERROR_H

#define MIN_PERCENTAGE 0.99f
#define MAX_PERCENTAGE 1.01f

void check_relative_error_gpu(double *array, float *array_rp, int N);
void calc_relative_error_gpu(double *array, float *array_rp, float *err_out, int N);
float find_max_relative_error_gpu(double *d_array, float *d_array_rp, int N, float *h_err, float *d_err);
unsigned long long get_dmr_error();

#endif