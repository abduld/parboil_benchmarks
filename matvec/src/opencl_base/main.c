/***************************************************************************
 *cr
 *cr            (C) Copyright 2008-2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <parboil.h>

static void write_dummy(char* file) {
  FILE* fp = fopen(file, "wb");
  fclose(fp);
}

extern void create_cpu_data(float** output, float** mat, float** vec, int nRows, int nCols);
extern void matvec_cpu(float* output, float* mat, float* vec, int nRows, int nCols);
extern int matvec_gpu(float* output, float* mat, float* vec, int nRows, int nCols, int nIter, struct pb_Parameters *parameters, struct pb_TimerSet* timers);

static void debug(float* data, int nRows, int nCols) {
  int col, row;
  for (col = 0; col < nCols; col++) {
    for (row = 0; row < nRows; row++) {
      int idx = col * nCols + row;
      printf ("%2.1f\t", data[idx]);
    }
    printf ("\n");
  }
  printf ("\n");
}

int main(int argc, char *argv[]) {
  struct pb_Parameters *parameters;
  struct pb_TimerSet timers;

  /* Read input parameters */
  parameters = pb_ReadParameters(&argc, argv);
  if (parameters == NULL) {
    fprintf(stderr, "Error in parameters.\n");
    exit(1);
  }

  if (argc < 3) {
    fprintf(stderr, "Not enough parameters. Usage: <binary> <nRows> <nCols> <nIter>\n");
    exit(1);
  }

  pb_InitializeTimerSet(&timers);

  int arg1 = atoi(argv[1]);
  int arg2 = atoi(argv[2]);
  int arg3 = atoi(argv[3]);

  int nRows = arg1;
  int nCols = arg2;
  int nIter = arg3;

  float* output;
  float* mat;
  float* vec;

  pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);
  create_cpu_data(&output, &mat, &vec, nRows, nCols);
  matvec_cpu(output, mat, vec, nRows, nCols);
  int result = matvec_gpu(output, mat, vec, nRows, nCols, nIter, parameters, &timers);
  pb_SwitchToTimer(&timers, pb_TimerID_IO);
  if (parameters->outFile) {
    write_dummy(parameters->outFile);
  }

  pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);

  pb_SwitchToTimer(&timers, pb_TimerID_NONE);
  pb_PrintTimerSet(&timers);
  pb_FreeParameters(parameters);

  return result;
}
