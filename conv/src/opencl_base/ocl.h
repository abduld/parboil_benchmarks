#ifndef __OCLH__
#define __OCLH__

char* readFile(const char*);

#define CHECK_ERROR(errorMessage)           \
  if(clStatus != CL_SUCCESS)                \
  {                                         \
     printf("Error: %s (%d)!\n",errorMessage, clStatus);   \
     printf("Line: %d\n",__LINE__);         \
     exit(1);                               \
  }

#endif
