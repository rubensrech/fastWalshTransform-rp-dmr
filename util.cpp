#include "util.h"

#include <stdlib.h>

unsigned int log2_host(unsigned int n) {
    return (n > 1) ? 1 + log2(n >> 1) : 0;
}

float find_max_i(float *array, int N) {
    int iMax = 0;
    for (int i = 1; i < N; i++)
        if (array[i] > array[iMax]) iMax = i;
    return iMax;
}

// > Timing functions

void getTimeNow(Time *t) {
    gettimeofday(t, 0);
}

double elapsedTime(Time t1, Time t2) {
    return (1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec)/1000.0;
}

// > Arguments funcations

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


// > Input/Output functions

bool save_input(double *data, int dataN, double *kernel, int kernelN, char *filename) {
    FILE *f = fopen(filename, "wb");
    if (f == NULL) {
        fprintf(stderr, "ERROR: could not save input to file\n");
        exit(-1);
    }

    fwrite(&dataN, sizeof(int),    1,     f);
    fwrite(data,   sizeof(double), dataN, f);

    fwrite(&kernelN, sizeof(int),    1,       f);
    fwrite(kernel,   sizeof(double), kernelN, f);

    fclose(f);
    return true;
}

bool save_input(double *data, int dataN, double *kernel, int kernelN, float maxErr) {
    char filename[100];
    snprintf(filename, 100, "input-%1.3f.data", maxErr);
    return save_input(data, dataN, kernel, kernelN, filename);
}

bool save_input(double *data, int dataN, double *kernel, int kernelN, int maxErrBit) {
    char filename[100];
    snprintf(filename, 100, "input-bit-%d.data", maxErrBit);
    return save_input(data, dataN, kernel, kernelN, filename);
}

bool load_input(char *filename, double *data, int dataN, double *kernel, int kernelN) {
    int dataN_, kernelN_;

    FILE *f = fopen(filename, "rb");
    if (f == NULL) {
        fprintf(stderr, "ERROR: could not read input from file\n");
        exit(-1);
    }

    fread(&dataN_, sizeof(int),    1,      f);
    if (dataN != dataN_) {
        fprintf(stderr, "ERROR: Input data doesn't match the expected size\n");
        exit(-1);
    }

    fread(data,  sizeof(double), dataN, f);

    fread(&kernelN_, sizeof(int),    1,        f);
    if (kernelN != kernelN_) {
        fprintf(stderr, "ERROR: Input data doesn't match the expected size\n");
        exit(-1);
    }

    fread(kernel,  sizeof(double), kernelN, f);

    return true;
}

bool save_output(double *output, int N, float maxErr) {
    char filename[100];
    snprintf(filename, 100, "output-%1.3f.data", maxErr);

    FILE *f = fopen(filename, "wb");
    if (f == NULL) {
        fprintf(stderr, "ERROR: could not save output to file\n");
        exit(-1);
    }

    fwrite(&N,     sizeof(int),    1, f);
    fwrite(output, sizeof(double), N, f);

    fclose(f);
    return true;
}

bool save_output(double *output, int N) {
    FILE *f = fopen("output.data", "wb");
    if (f == NULL) {
        fprintf(stderr, "ERROR: could not save output to file\n");
        exit(-1);
    }

    fwrite(&N,     sizeof(int),    1, f);
    fwrite(output, sizeof(double), N, f);

    fclose(f);
    return true;
}

bool compare_output_with_golden(double *output, int N, const char *filename) {
    FILE *f = fopen(filename, "rb");
    double *gold_output;
    int n, i;

    if (f == NULL) {
        fprintf(stderr, "ERROR: could not read output from file\n");
        exit(-1);
    }

    fread(&n, sizeof(int), 1, f);
    if (n != N) {
        fprintf(stderr, "ERROR: Output data doesn't match the expected size\n");
        exit(-1);
    }

    gold_output = (double*)malloc(N * sizeof(double));
    fread(gold_output, sizeof(double), N, f);

    bool outputsMatch = true;
    int countDiffs = 0;
    double minAbsErr = __DBL_MAX__, maxAbsErr = __DBL_MIN__, avgAbsErr = 0;
    double minRelErr = __DBL_MAX__, maxRelErr = __DBL_MIN__, avgRelErr = 0;
    uint32_t bitErrs[64] = {0};
    for (i = 0; i < N; i++) {
        if (output[i] != gold_output[i]) {
            outputsMatch = false;
            countDiffs++;

            // UINT diff
            uint32_t outData = *((uint32_t*) &(output[i]));
            uint32_t goldData = *((uint32_t*) &(gold_output[i]));
            uint32_t uintErr = SUB_ABS(outData, goldData);
            int errBit = log2_host(uintErr);
            bitErrs[errBit]++;
            // Abs diff
            double absErr = SUB_ABS(output[i], gold_output[i]);
            if (absErr > maxAbsErr) maxAbsErr = absErr;
            if (absErr < minAbsErr) minAbsErr = absErr;
            avgAbsErr += absErr;
            // Rel err
            double relErr = abs(1 - output[i]/gold_output[i]);
            if (relErr > maxRelErr) maxRelErr = relErr;
            if (relErr < minRelErr) minRelErr = relErr;
            avgRelErr += relErr;
        }
    }
    if (countDiffs > 0) {
        avgAbsErr /= countDiffs;
        avgRelErr /= countDiffs;
    } else {
        avgAbsErr = minAbsErr = maxAbsErr = 0;
        avgRelErr = minRelErr = maxRelErr = 0;
    }
    // Save stats to file
    FILE *fstats = fopen("out-vs-gold-stats.txt", "w");
    fprintf(fstats, "===================================================\n");
    fprintf(fstats, "========== Output VS Golden output stats ==========\n");
    fprintf(fstats, "===================================================\n");
    fprintf(fstats, "  > Outputs match: %s\n", outputsMatch ? "YES" : "NO");
    fprintf(fstats, "  > Number of diff values: %u\n", countDiffs);
    fprintf(fstats, "\n");
    fprintf(fstats, "  > Max absolute err: %.20e\n", maxAbsErr);
    fprintf(fstats, "  > Min absolute err: %.20e\n", minAbsErr);
    fprintf(fstats, "  > Avg absolute err: %.20e\n", avgAbsErr);
    fprintf(fstats, "\n");
    fprintf(fstats, "  > Max relative err: %.20e\n", maxRelErr);
    fprintf(fstats, "  > Min relative err: %.20e\n", minRelErr);
    fprintf(fstats, "  > Avg relative err: %.20e\n", avgRelErr);
    fprintf(fstats, "\n");
    fprintf(fstats, "  > Number of errors in each bit:\n");
    for (i = 0; i < 64; i++) {
        fprintf(fstats, "    * Bit %2u: %u\n", i, bitErrs[i]);
    }

    fclose(f);
    fclose(fstats);
    return outputsMatch;
}