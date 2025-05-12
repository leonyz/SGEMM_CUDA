#pragma once

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

template <const int BLOCKSIZE>
__global__ void sgemm_shared_mem_block(int M, int N, int K, float alpha,
                                       const float *A, const float *B,
                                       float beta, float *C) {
    // the grid and block dims are the same as Kernel 2.
    // we'll use the same mapping of threads to entries of C
    assert(blockDim.x == BLOCKSIZE * BLOCKSIZE);
    const uint C_x = blockIdx.x * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);
    const uint C_y = blockIdx.y * BLOCKSIZE + (threadIdx.x % BLOCKSIZE);

    // in this kernel though (and maybe it's generally good practice), we
    // can advance the pointers of A and B to simplify our arithmetic
    A = A + blockIdx.x * BLOCKSIZE * K;
    B = B + blockIdx.y * BLOCKSIZE;

    // initialize the SMEM arrays
  __shared__ float As[BLOCKSIZE * BLOCKSIZE];
  __shared__ float Bs[BLOCKSIZE * BLOCKSIZE];

  if (C_x < M && C_y < N) {
    float out = 0.0;

    const uint thread_row = threadIdx.x / BLOCKSIZE;
    const uint thread_col = threadIdx.x % BLOCKSIZE;

    // I don't want to deal with the case where BLOCKSIZE 
    // doesn't divide the accumulation dim. so assert against it.
    assert(K % BLOCKSIZE == 0);
    for (int i = 0; i * BLOCKSIZE < K; i++) {
      // step 0: load a BLOCKSIZE x BLOCKSIZE chunk of A and B into As and Bs
      //         since we're already shifted the A and B pointers to the right
      //         starting positions, we just need to handle the striding across rows
      const uint load_A = thread_row * K + thread_col;
      const uint load_B = thread_row * N + thread_col;
      As[thread_row * BLOCKSIZE + thread_col] = A[load_A];
      Bs[thread_row * BLOCKSIZE + thread_col] = B[load_B];

      // block to make sure all entries are loaded to SMEM before doing arithmetic
      __syncthreads();
    
      // step 1: compute out as much as possible
      for (int j = 0; j < BLOCKSIZE; j++) {
        out += As[thread_row * BLOCKSIZE + j] * Bs[j * BLOCKSIZE + thread_col];
      }

      // block to make sure all entries are loaded to SMEM before advancing pointers
      __syncthreads();

      // step 2: advance the A and B pointers
      A = A + BLOCKSIZE;
      B = B + BLOCKSIZE * N;    
    }
    C[C_x * N + C_y] = C[C_x * N + C_y] * beta + out * alpha;
  }
    
}