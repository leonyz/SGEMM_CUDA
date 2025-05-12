#pragma once

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

template <const uint BLOCKSIZE>
__global__ void sgemm_global_mem_coalesce(int M, int N, int K, float alpha,
                                          const float *A, const float *B,
                                          float beta, float *C) {
  // note that compared to kernel 1, the blocks are 1D of size 32x32 instead of (32, 32)
  // I think this only works if the number of threads in a block is the square of BLOCKSIZE.

  assert(blockDim.x == BLOCKSIZE*BLOCKSIZE);
  const uint x = blockIdx.x * BLOCKSIZE + threadIdx.x / BLOCKSIZE;
  const uint y = blockIdx.y * BLOCKSIZE + threadIdx.x % BLOCKSIZE;
  // with the above indexing structure, as threadIdx.x increments, the x-coordinate 
  // only changes every BLOCKSIZE times while the y-coordinate increments every time

  if (x < M and y < N) {
    float out = 0.0;
    for (int i = 0; i < K; i++) {
      const uint A_idx = x * K + i;
      const uint B_idx = i * N + y;
      out += A[A_idx] * B[B_idx];
    }
    const uint C_idx = x * N + y;
    C[C_idx] = alpha * out + beta * C[C_idx];
  }   
}