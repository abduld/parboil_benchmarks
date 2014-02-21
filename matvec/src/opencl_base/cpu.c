#include <stdio.h>

void create_cpu_data(float** output, float** mat, float** vec, int nRows, int nCols) {
  *mat = (float *) malloc(sizeof(float) * nRows * nCols);
  *vec = (float *) malloc(sizeof(float) * nCols);
  *output = (float *) malloc(sizeof(float) * nCols);
  // random data here
  int i, col, row;
  int idx = 0;
  for (col = 0; col < nCols; col++) {
    (*vec)[col] = col + 1;
    for (row = 0; row < nRows; row++) {
      (*mat)[idx] = idx % 4 * (row % 2 ? 1.0f : -1.0f);
    }
  }
}

void matvec_cpu(float* output, float* mat, float* vec, int nRows, int nCols) {
  int row;
  int col;
  for (row = 0; row < nRows; row++) {
    float result = 0.0f;
    for (col = 0; col < nCols; col++) {
      result += mat[col * nRows + row] * vec[col];
    }
    output[row] = result;
  }
}

