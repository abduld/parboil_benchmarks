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

extern void create_cpu_data(float** output, float** input, float** kernel, int half_kernel, int image_height, int image_width);
extern void conv_cpu(float* output, float* input, float* kernel, int half_kernel, int image_height, int image_width);
extern int conv_gpu(float* output, float* input, float* kernel, int half_kernel, int image_height, int image_width, int nIter, struct pb_Parameters *parameters, struct pb_TimerSet* timers);

static void debug(float* data, int image_height, int image_width) {
  int r, c;
  for (r = 0; r < image_height; r++) {
    for (c = 0; c < image_width; c++) {
      int idx = c + r * image_height;
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
    fprintf(stderr, "Not enough parameters. Usage: <binary> <half kernel dim> <image dim>\n");
    exit(1);
  }

  pb_InitializeTimerSet(&timers);

  int arg1 = atoi(argv[1]);
  int arg2 = atoi(argv[2]);
  int arg3 = atoi(argv[3]);

  int half_filter_size = arg1;
  int fsz = (2 * half_filter_size + 1);
  int image_w = arg2;
  int image_h = arg2;
  int nIter = arg3;

  float* output;
  float* input;
  float* filter;

  pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);
  create_cpu_data(&output, &input, &filter, half_filter_size, image_h, image_w);
  conv_cpu(output, input, filter, half_filter_size, image_h, image_w);
  int result = conv_gpu(output, input, filter, half_filter_size, image_h, image_w, nIter, parameters, &timers);
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
