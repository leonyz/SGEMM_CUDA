#pragma once

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void __launch_bounds__((BM * BN) / (TM * TN), 1)
    sgemm2DBlocktiling(int M, int N, int K, float alpha, const float *A,
                       const float *B, float beta, float *C) {
    // advance the pointers for A and B to the right block
    const uint A_row = blockIdx.y * BM;
    const uint B_col = blockIdx.x * BN;
    A = A + A_row * K;
    B = B + B_col;

    // we assume each thread calculates TM * TN entries of C
    assert(blockDim.x == BM * BN / TM / TN);

    // calculate the upper-left corner of the 2D blocktile within the block
    // that our particular thread is computing
    const uint thread_row_tmp = threadIdx.x / (BN / TN);
    const uint thread_col_tmp = threadIdx.x % (BN / TN);
    const uint thread_row = thread_row_tmp * TM;
    const uint thread_col = thread_col_tmp * TN;

    // advance the pointer for C to the right block, then the right thread
    C = C + A_row * N + B_col;
    C = C + thread_row * N + thread_col;

    // calculate the upper-left corner of the chunk of A and B that our particular
    // thread is loading into As and Bs
    const uint thread_As_entries_to_load = BM * BK / blockDim.x;
    const uint thread_As_load_row = threadIdx.x / (BK / thread_As_entries_to_load);
    const uint thread_As_load_col_tmp = threadIdx.x % (BK / thread_As_entries_to_load);
    const uint thread_As_load_col = thread_As_load_col_tmp * thread_As_entries_to_load;

    const uint thread_Bs_entries_to_load = BK * BN / blockDim.x;
    const uint thread_Bs_load_row = threadIdx.x / (BN / thread_Bs_entries_to_load);
    const uint thread_Bs_load_col_tmp = threadIdx.x % (BN / thread_Bs_entries_to_load);
    const uint thread_Bs_load_col = thread_Bs_load_col_tmp * thread_Bs_entries_to_load;

    // initialize As and Bs
    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    // initialize the parts of As and Bs we'll use for computing entries of C
    float A_local[TM * BK] = {0.0};
    float B_local[BK * TN] = {0.0};

    // initialize the running reductions for C
    float C_out[TM*TN] = {0.0};

    if ((A_row + thread_row < M) && (B_col + thread_col < N)) {
      for (int i = 0; i * BK < K; i++) {
        // step 0: load from A and B into As and Bs
        for (int load_idx = 0; load_idx < thread_As_entries_to_load; load_idx++) {
          As[thread_As_load_row * BK + thread_As_load_col + load_idx] = A[thread_As_load_row * K + thread_As_load_col + load_idx];
        }
        for (int load_idx = 0; load_idx < thread_Bs_entries_to_load; load_idx++) {
          Bs[thread_Bs_load_row * BN + thread_Bs_load_col + load_idx] = B[thread_Bs_load_row * N + thread_Bs_load_col + load_idx];
        }

        __syncthreads();

        // step 1: load from As and Bs into A_local and B_local
        for (int BK_idx = 0; BK_idx < BK; BK_idx++) {
          for (int A_local_row = 0; A_local_row < TM; A_local_row++) {
            A_local[A_local_row * BK + BK_idx] = As[thread_row * BK + A_local_row * BK + BK_idx];
          }
          for (int B_local_col = 0; B_local_col < TN; B_local_col++) {
            B_local[BK_idx * TN + B_local_col] = Bs[thread_col + BK_idx * BN + B_local_col];
          }
        }

        // step 2: compute C_out reduction for this blocktile
        for (int C_out_row = 0; C_out_row < TM; C_out_row++) {
          for (int C_out_col = 0; C_out_col < TN; C_out_col++) {
            for (int BK_idx = 0; BK_idx < BK; BK_idx++) {
              C_out[C_out_row * TN + C_out_col] += A_local[C_out_row * BK + BK_idx] * B_local[BK_idx * TN + C_out_col];
            }
          }
        }

        __syncthreads();
        // step 3: advance pointers
        A = A + BK;
        B = B + BK * N;
      }

      for (int C_out_row = 0; C_out_row < TM; C_out_row++) {
        for (int C_out_col = 0; C_out_col < TN; C_out_col++) {
          C[C_out_row * N + C_out_col] = C_out[C_out_row * TN + C_out_col] * alpha + C[C_out_row * N + C_out_col] * beta;
        }
      }
    }
}