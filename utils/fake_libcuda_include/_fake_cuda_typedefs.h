#ifndef _FAKE_CUDA_TYPEDEFS_H
#define _FAKE_CUDA_TYPEDEFS_H

struct uint3
{
  unsigned int x, y, z;
};

struct dim3
{
    unsigned int x, y, z;
};

enum cudaMemcpyKind
{
  cudaMemcpyHostToHost          =   0,      /**< Host   -> Host */
  cudaMemcpyHostToDevice        =   1,      /**< Host   -> Device */
  cudaMemcpyDeviceToHost        =   2,      /**< Device -> Host */
  cudaMemcpyDeviceToDevice      =   3,      /**< Device -> Device */
  cudaMemcpyDefault             =   4       /**< Default based unified virtual address space */
};

typedef struct dim3 dim3;
typedef struct uint3 uint3;

uint3 extern const threadIdx;
uint3 extern const blockIdx;

dim3 extern const blockDim;
dim3 extern const gridDim;
