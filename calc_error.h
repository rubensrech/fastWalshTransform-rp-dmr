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

unsigned long long get_zeros_fp64();
unsigned long long get_zeros_fp32();
unsigned long long get_negatives();
unsigned long long get_zeros_diff_gt_non_zeros_thresh();
unsigned int get_max_diff_zeros_double_val();
unsigned int get_max_diff_zeros_float_val();
unsigned int get_max_diff_non_zeros_double_val();
unsigned int get_max_diff_non_zeros_float_val();
void get_diffs_distribution(unsigned long long *dist);

void calc_relative_error_gpu(double *array, float *array_rp, float *err_out, int N);

void find_max_relative_and_abs_error_gpu(double *array, float *array_rp, int N);
void calc_error_hybrid_gpu(double *array, float *array_rp, int N);

float get_max_rel_error();
float get_max_rel_error_double_val();
float get_max_rel_error_float_val();
float get_max_rel_error_uint_err();

float get_max_abs_error();
float get_max_abs_error_double_val();
float get_max_abs_error_float_val();
float get_max_abs_error_uint_err();

#endif