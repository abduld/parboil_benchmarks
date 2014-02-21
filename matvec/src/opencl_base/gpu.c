/***************************************************************************
 *cr
 *cr            (C) Copyright 2008-2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/

#include <CL/cl.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <parboil.h>

#include "ocl.h"

int matvec_gpu(float* output, float* mat, float* vec, int nRows, int nCols, int nIter, struct pb_Parameters *parameters, struct pb_TimerSet* timers) {
  pb_Context* pb_context;
  pb_context = pb_InitOpenCLContext(parameters);
  if (pb_context == NULL) {
    fprintf (stderr, "Error: No OpenCL platform/device can be found.");
    return;
  }

  cl_int clStatus;
  cl_device_id clDevice = (cl_device_id) pb_context->clDeviceId;
  cl_platform_id clPlatform = (cl_platform_id) pb_context->clPlatformId;
  cl_context clContext = (cl_context) pb_context->clContext;

  cl_command_queue clCommandQueue = clCreateCommandQueue(clContext,clDevice,CL_QUEUE_PROFILING_ENABLE,&clStatus);
  CHECK_ERROR("clCreateCommandQueue")

  pb_SetOpenCL(&clContext, &clCommandQueue);

  const char* clSource[] = {readFile("src/opencl_base/kernel.cl")};
  cl_program clProgram = clCreateProgramWithSource(clContext,1,clSource,NULL,&clStatus);
  CHECK_ERROR("clCreateProgramWithSource")

  char clOptions[50];
  sprintf(clOptions,"-I src/opencl_base");  //-cl-nv-verbose

  clStatus = clBuildProgram(clProgram,1,&clDevice,clOptions,NULL,NULL);
  if (clStatus != CL_SUCCESS) {
    size_t string_size = 0;
    clGetProgramBuildInfo(clProgram, clDevice, CL_PROGRAM_BUILD_LOG,
                          0, NULL, &string_size);
    char* string = malloc(string_size*sizeof(char));
    clGetProgramBuildInfo(clProgram, clDevice, CL_PROGRAM_BUILD_LOG,
                          string_size, string, NULL);
    puts(string);
  }

  CHECK_ERROR("clBuildProgram")

  cl_kernel clKernel = clCreateKernel(clProgram,"matvec",&clStatus);
  CHECK_ERROR("clCreateKernel")

  float* output_gpu;
  cl_mem output_d;
  cl_mem mat_d;
  cl_mem vec_d;
  int image_sz = nCols * nRows;
  int vec_sz = nCols;
  output_gpu = (float *) malloc(sizeof(float) * image_sz);
  output_d = clCreateBuffer(clContext,CL_MEM_WRITE_ONLY,vec_sz*sizeof(float),NULL,&clStatus);
  CHECK_ERROR("clCreateBuffer")
  mat_d  = clCreateBuffer(clContext,CL_MEM_READ_ONLY,image_sz*sizeof(float),NULL,&clStatus);
  CHECK_ERROR("clCreateBuffer")
  vec_d  = clCreateBuffer(clContext,CL_MEM_READ_ONLY,vec_sz*sizeof(float),NULL,&clStatus);
  CHECK_ERROR("clCreateBuffer")

  pb_SwitchToTimer(timers, pb_TimerID_COPY);
  clStatus = clEnqueueWriteBuffer(clCommandQueue,mat_d,CL_TRUE,0,image_sz*sizeof(float),mat,0,NULL,NULL);
  CHECK_ERROR("clEnqueueWriteBuffer")
  clStatus = clEnqueueWriteBuffer(clCommandQueue,vec_d,CL_TRUE,0,vec_sz*sizeof(float),vec,0,NULL,NULL);
  CHECK_ERROR("clEnqueueWriteBuffer")

  pb_SwitchToTimer(timers, pb_TimerID_COMPUTE);
  clStatus = clSetKernelArg(clKernel, 0, sizeof(cl_mem), &output_d);
  clStatus |= clSetKernelArg(clKernel, 1, sizeof(cl_mem), &mat_d);
  clStatus |= clSetKernelArg(clKernel, 2, sizeof(cl_mem), &vec_d);
  clStatus |= clSetKernelArg(clKernel, 3, sizeof(int),    &nRows);
  clStatus |= clSetKernelArg(clKernel, 4, sizeof(int),    &nCols);
  CHECK_ERROR("clSetKernelArg")

  /* loop over z-dimension, invoke OpenCL kernel for each x-y plane */
  size_t blockDim[1] = { 512 };
  size_t gridDim[1] = {
    ((nRows + blockDim[0] - 1) / blockDim[0]) * blockDim[0]
  };
  pb_SwitchToTimer(timers, pb_TimerID_KERNEL);
  { // for just repeat.
    int i;
    for (i = 0; i < nIter; i++) {
      clStatus = clEnqueueNDRangeKernel(clCommandQueue,clKernel,1,NULL,gridDim,blockDim,0,NULL,NULL);
      CHECK_ERROR("clEnqueueNDRangeKernel")
    }
  }
  clStatus = clFinish(clCommandQueue);
  CHECK_ERROR("clFinish")

  /* copy result regions from OpenCL device */
  pb_SwitchToTimer(timers, pb_TimerID_COPY);
  clStatus = clEnqueueReadBuffer(clCommandQueue,output_d,CL_TRUE,0,vec_sz*sizeof(float),output_gpu,0,NULL,NULL);
  CHECK_ERROR("clEnqueueReadBuffer")

  pb_SwitchToTimer(timers, pb_TimerID_COMPUTE);

  /* free OpenCL memory allocations */
  clStatus = clReleaseMemObject(vec_d);
  clStatus = clReleaseMemObject(output_d);
  clStatus = clReleaseMemObject(mat_d);
  CHECK_ERROR("clReleaseMemObject")

  clStatus = clReleaseKernel(clKernel);
  clStatus = clReleaseProgram(clProgram);
  clStatus = clReleaseCommandQueue(clCommandQueue);
  clStatus = clReleaseContext(clContext);

  free((void*)clSource[0]);

  /* compuare results */
  int retval = 0;
  {
    int i;
    for (i = 0; i < nCols; i++) {
      float delta = output_gpu[i] - output[i];
      if (delta != 0.0f) {
        printf ("Results are different at %d: %f vs %f\n", i, output_gpu[i], output[i]);
        printf ("Mismatch\n");
        retval = 1;
        break;
      }
    }
  }
  free(output_gpu);
  return retval;
}

