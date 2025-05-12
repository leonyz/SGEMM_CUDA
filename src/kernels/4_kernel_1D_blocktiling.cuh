#pragma once

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

template <const int BM, const int BN, const int BK, const int TM>
__global__ void sgemm1DBlocktiling(int M, int N, int K, float alpha,
                                   const float *A, const float *B, float beta,
                                   float *C) {
    // advance the pointers for A and B
    const uint A_row = blockIdx.y * BM * K;
    const uint B_col = blockIdx.x * BN;
    A = A + A_row;
    B = B + B_col;

    // initialize As and Bs
    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    // calculate the block shift and thread-specific start row and col idxs
    const uint C_row = blockIdx.y * BM * N;
    const uint C_col = blockIdx.x * BN;
    const uint C_thread_row_start = threadIdx.x / BN;
    const uint C_thread_row = C_thread_row_start * TM;
    const uint C_thread_col = threadIdx.x % BN;
    float thread_out[TM] = {0.0};

    for (int i = 0; i * BK < K; i++) {
      // step 0: fill in As and Bs. we assume one thread fills in one entry of each 
      // and protect with an assert.
      assert(BM * BK / BN / TM == 1);
      const uint As_row = threadIdx.x / BK;
      const uint As_col = threadIdx.x % BK;

      const uint Bs_row = threadIdx.x / BN;
      const uint Bs_col = threadIdx.x % BN;

      As[As_row * BK + As_col] = A[As_row * K + As_col];
      Bs[Bs_row * BN + Bs_col] = B[Bs_row * N + Bs_col];

      __syncthreads();

      // step 1: compute as much of thread_out as possible.
      for (int out_row = 0; out_row < TM; out_row++) {
        for (int j = 0; j < BK; j++) {
          thread_out[out_row] += As[C_thread_row * BK + out_row * BK + j] * Bs[C_thread_col + j * BN];
        }
      }
      __syncthreads();

      // step 2: advance pointers for A and B
      A = A + BK;
      B = B + BK * N;
    }
    // advance the pointer for C
    C = C + C_row + C_col;
    for (int out_row = 0; out_row < TM; out_row++) {
      C[C_thread_row * N + out_row * N + C_thread_col] = C[C_thread_row * N + out_row * N + C_thread_col] * beta + alpha * thread_out[out_row];
    }
    
  }
