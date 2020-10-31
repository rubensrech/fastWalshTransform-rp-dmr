#include "util.h"

#include <stdlib.h>

float find_max(float *array, int N) {
    float max = array[0];
    for (int i = 1; i < N; i++)
        if (array[i] > max) max = array[i];
    return max;
}

bool save_input(double *data, int dataN, double *kernel, int kernelN, float maxErr) {
    char filename[100];
    snprintf(filename, 100, "input-%1.3f.data", maxErr);
    FILE *f = fopen(filename, "wb");
    if (f == NULL) {
        return false;
    }

    fwrite(&dataN, sizeof(int),    1,     f);
    fwrite(data,   sizeof(double), dataN, f);

    fwrite(&kernelN, sizeof(int),    1,       f);
    fwrite(kernel,   sizeof(double), kernelN, f);

    fclose(f);
    return true;
}