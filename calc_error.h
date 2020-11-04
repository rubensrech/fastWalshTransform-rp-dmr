#ifndef CALC_ERROR_H
#define CALC_ERROR_H

#define REL_ERROR    0
#define UINT_ERROR   1

#define MIN_PERCENTAGE 0.82f
#define MAX_PERCENTAGE 1.18f

#define UINT_THRESHOLD 0

#define SUB_ABS(lhs, rhs) ((lhs > rhs) ? (lhs - rhs) : (rhs - lhs))

void check_error_gpu(double *array, float *array_rp, int N);
unsigned long long get_dmr_error();

void find_max_uint_error_gpu(double *array, float *array_rp, int N) ;
unsigned int get_max_uint_error_non_zeros();
unsigned int get_max_uint_error_zeros();

void calc_relative_error_gpu(double *array, float *array_rp, float *err_out, int N);

#endif