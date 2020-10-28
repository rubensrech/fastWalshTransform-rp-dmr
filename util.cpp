#include "util.h"

float find_max(float *array, int N) {
    float max = array[0];
    for (int i = 1; i < N; i++)
        if (array[i] > max) max = array[i];
    return max;
}