#pragma once

#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

/*

Matrix sizes:
MxK * KxN = MxN

*/

// this kernel assigns the computation of an entry of C to one thread. 
// If C is 4096 x 4096, we'll need 4096 x 4096 threads.
// With a block size of 32, that means we'll need we'll need a 128 x 128 grid of blocks.
__global__ void sgemm_naive(int M, int N, int K, float alpha, const float *A,
                            const float *B, float beta, float *C) {
  const uint x = blockIdx.x * blockDim.x + threadIdx.x;
  const uint y = blockIdx.y * blockDim.y + threadIdx.y;

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