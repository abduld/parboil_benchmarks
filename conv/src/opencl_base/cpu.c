#include <stdio.h>

void create_cpu_data(float** output, float** input, float** kernel, int half_kernel, int image_height, int image_width) {
  int kernel_sz = (2*half_kernel + 1) * (2*half_kernel + 1);
  *kernel = (float *) malloc(sizeof(float) * kernel_sz);
  *input = (float *) malloc(sizeof(float) * image_width * image_height);
  *output = (float *) malloc(sizeof(float) * image_width * image_height);
  // random data here
  int i, r, c;
  for (i = 0; i < kernel_sz; i++) {
    (*kernel)[i] = i;
  }
  for (r = 0; r < image_height; r++) {
    for (c = 0; c < image_width; c++) {
      int idx  = r*image_width+c;
      (*input)[idx] = idx % 4 * (c % 2 ? 1.0f : -1.0f);
    }
  }
}

void conv_cpu(float* output, float* input, float* filter, int HALF_FILTER_SIZE, int IMAGE_H, int IMAGE_W) {

  int row;
  int col;
  for (row = 0; row < IMAGE_H; row++) {
    for (col = 0; col < IMAGE_W; col++) {
      int idx = row * IMAGE_W + col;
      if (col < HALF_FILTER_SIZE ||
          col > IMAGE_W - HALF_FILTER_SIZE - 1 ||
          row < HALF_FILTER_SIZE ||
          row > IMAGE_H - HALF_FILTER_SIZE - 1) {
        output[idx] = 0.0f;
      } else {
        // perform convolution
        int fIndex = 0;
        float result = 0.0f;

        int r;
        for (r = -HALF_FILTER_SIZE; r <= HALF_FILTER_SIZE; r++) {
          int curRow = idx + r * IMAGE_W;
          int c;
          for (c = -HALF_FILTER_SIZE; c <= HALF_FILTER_SIZE; c++) {
            int offset = c;
            result += input[ curRow + offset ] * filter[fIndex];
            fIndex++;
          }
        }
        output[idx] = result;
      }
    }
  }
}

