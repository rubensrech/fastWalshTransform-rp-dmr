#include "util.h"

#include <stdlib.h>

void del_arg(int argc, char **argv, int index) {
    int i;
    for(i = index; i < argc-1; ++i) argv[i] = argv[i+1];
    argv[i] = 0;
}

int find_int_arg(int argc, char **argv, char *arg, int def) {
    int i;
    for (i = 0; i < argc-1; ++i) {
        if(!argv[i]) continue;
        if (0==strcmp(argv[i], arg)) {
            def = atoi(argv[i+1]);
            del_arg(argc, argv, i);
            del_arg(argc, argv, i);
            break;
        }
    }
    return def;
}

char *find_char_arg(int argc, char **argv, char *arg, char *def) {
    int i;
    for(i = 0; i < argc-1; ++i){
        if(!argv[i]) continue;
        if(0==strcmp(argv[i], arg)){
            def = argv[i+1];
            del_arg(argc, argv, i);
            del_arg(argc, argv, i);
            break;
        }
    }
    return def;
}

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
    if (f == NULL) return false;

    fwrite(&dataN, sizeof(int),    1,     f);
    fwrite(data,   sizeof(double), dataN, f);

    fwrite(&kernelN, sizeof(int),    1,       f);
    fwrite(kernel,   sizeof(double), kernelN, f);

    fclose(f);
    return true;
}

bool load_input(char *filename, double *data, int *dataN, double *kernel, int *kernelN) {
    FILE *f = fopen(filename, "rb");
    if (f == NULL) return false;

    fread(dataN, sizeof(int),    1,      f);
    fread(data,  sizeof(double), *dataN, f);

    fread(kernelN, sizeof(int),    1,        f);
    fread(kernel,  sizeof(double), *kernelN, f);

    return true;
}

bool save_output(double *output, int N, float maxErr) {
    char filename[100];
    snprintf(filename, 100, "output-%1.3f.data", maxErr);
    FILE *f = fopen(filename, "wb");
    if (f == NULL) return false;

    fwrite(&N,     sizeof(int),    1, f);
    fwrite(output, sizeof(double), N, f);

    fclose(f);
    return true;
}

bool save_output(double *output, int N) {
    FILE *f = fopen("output.data", "wb");
    if (f == NULL) return false;

    fwrite(&N,     sizeof(int),    1, f);
    fwrite(output, sizeof(double), N, f);

    fclose(f);
    return true;
}