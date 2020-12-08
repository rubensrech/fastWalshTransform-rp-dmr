#ifndef CALC_ERROR_H
#define CALC_ERROR_H

#define REL_ERROR    0
#define UINT_ERROR   1
#define HYBRID       2

#define ABS_ERR_THRESHOLD   0.000031f   // Max ABS error(input-bit-21.data) = 0.000031
#define REL_ERR_THRESHOLD   0.000001f   // Max REL error(input-bit-21.data) = 0.000001
#define UINT_THRESHOLD      13          // Max UINT error(input-bit-21.data) = 000...001101

#define MIN_PERCENTAGE 1.0f - REL_ERR_THRESHOLD
#define MAX_PERCENTAGE 1.0f + REL_ERR_THRESHOLD

#define ABS_ERR_UPPER_BOUND_VAL     0.02
#define IGNORE_VAL_FLAG             -999

////////////////////////////////////////////////////////////////////////////////
// Check error functions
////////////////////////////////////////////////////////////////////////////////

unsigned long long get_dmr_error();
void check_errors_gpu(double *array, float *array_rp, int N);

////////////////////////////////////////////////////////////////////////////////
// Find max error functions
////////////////////////////////////////////////////////////////////////////////

void calc_errors_gpu(double *array, float *array_rp, int N);
void get_diffs_distribution(unsigned long long *dist);
uint32_t get_max_uint_err();
void get_rel_error_array(float *relErrArr, int N);
void get_abs_error_array(float *absErrArr, int N);
uint32_t get_zeros_fp64();
uint32_t get_zeros_fp32();
uint32_t get_negatives();

#endif