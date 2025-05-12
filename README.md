##### Leon's local reminders

How to begin:
```
cd ~/cuda/SGEMM_CUDA
source ./.venv/bin/activate
cd build
```

To rebuild after code changes:
```
in ~/cuda/SGEMM_CUDA/build:
cmake .. && cmake --build .
```

To run a numbered kernel:
```
in ~/cuda/SGEMM_CUDA/build:
DEVICE=0 ./sgemm 1
```

To build in debug mode and run `cuda-gdb`:
```
cmake .. -DCMAKE_BUILD_TYPE=Debug && cmake --build .
DEVICE=0 cuda-gdb --args ./sgemm 3
break /home/leonz/cuda/SGEMM_CUDA/src/kernels/3_kernel_shared_mem_blocking.cuh:50
r
```
Inside the breakpoint, we can inspect variables and switch threads like so:
```
info cuda threads
cuda thread 5
print thread_row
print thread_col
```
To run an NSight profile:
```
in ~/cuda/SGEMM_CUDA:
make profile KERNEL=4
```
##### Introductory CUDA terminology

CUDA terminology:

* Each kernel creates a *grid*,
* Each grid has up-to-3 dimensions of *blocks*,
* Each block in turn has up-to-3 dimensions of *threads*,
* Each block has access to the same chunk of SMEM (shared memory)

The number of threads in a block is controlled by a vector usually called blockDim, and the number of blocks in a grid is defined by a vector usually called gridDim. These are provided in a call to the kernel so aren't defined in the kernel itself.

We can access the size of each block (in the number of threads) using `blockDim.x` and `blockDim.y` and `blockDim.z`. We can access the block and thread indices with e.g. `blockIdx.x` and `threadIdx.y` respectively.
##### Running tally of reimplemented kernel GFLOPs at 4096 x 4096 x 4096 on my laptop's RTX 4060:

* kernel 0 (CuBLAS): 9152.5 GFLOPs
* kernel 1 (naive): 115.0 GFLOPs
* kernel 2 (global memory coalescing): 813.3 GFLOPs
* kernel 3 (shared memory blocking): 1132.8 GFLOPs
* kernel 4 (1D block tiling): 3470.0 GFLOPs
* kernel 5 (2D block tiling): 7140.9 GFLOPs
##### Kernel 1 implementation notes

This one is pretty straightforward. We have our row and column of A & B for our relevant thread; we run through a for-loop to reduce along the K dim.

```
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
```

* kernel 1 (naive): 115.0 GFLOPs

Some napkin math: loading three 4096 x 4096 FP32 tensors and storing the 4096 x 4096 FP32 output takes 4 x 4096 x 4096 x 4 = 268MB of data to move on- and off-chip. The total FLOPs of the gemm is 4096 x 4096 x 4096 x 2 = 137 GFLOPs. My RTX 4060 gets 11.61 TFLOPs of FP32 compute and has 256 GBps of memory bandwidth, so at 100% FLOPs utilization this gemm would take 12ms and at 100% bandwidth utilization it would take 1ms, i.e. it should be mostly compute-bound if we do things right. Indeed, kernel 0 gets 9152.5 GFLOPs which is about 79% FLOPs utilization, good job CuBLAS!

Why is Kernel 1 so much slower than CuBLAS? If each thread were loading its own copy of the size-4096 row and output matrix vectors, this would require (2x4096+1) floats per thread, so 4096 x 4096 threads x (2x4096+1) floats loaded x 4 bytes per FP32 => 549GB. My RTX 4060 gets 256GBps, so I would expect this to take at least two seconds... actually it takes 1.19 seconds overall, probably because some optimizations are still kicking in to reduce memory accesses... e.g. some column vectors between consecutive threads are actually being shared.

> FOLLOW-UP: Figure out more exactly what's going on here. Run profiler?

tl;dr though: kernel 1 is memory bound. How do we make this faster? Share loads between threads. Summary:

* 1 warp = 32 threads. Contiguous memory accesses by threads in the same warp can be grouped together into a single 32B / 64B / 128B load (*global memory coalescing*)
* warps assigned to a warp scheduler, there are four warp schedulers per multiprocessor
* grouping of threads to warps happens based on consecutive threadIds, where threadId is defined as

```threadId = threadIdx.x+blockDim.x*(threadIdx.y+blockDim.y*threadIdx.z)```

AKA `x` first, `y` is scaled by `blockDim.x`, `z` is scaled by `blockDim.x*blockDim.y`.

Recall that blockDim is definable by the user / kernel call, so which threads end up in the same warp is to some extent affected by user decisions. The important thing for this next kernel is that sequential memory accesses by threads in the same warp can be grouped and executed by the scheduler as one memory access.
![[Pasted image 20250507231926.png]]
In Kernel 1, threads in the same warp shared the same column and saw non-contiguous row accesses. By adjusting the thread-to-entry-of-C assignment so that threads in the same warp travel along the rows of C, we can share rows between threads and have contiguous column accesses.

  ##### Kernel 2 implementation notes

```
template <const uint BLOCKSIZE>
__global__ void sgemm_global_mem_coalesce(int M, int N, int K, float alpha,
                                          const float *A, const float *B,
                                          float beta, float *C) {
  // note that compared to kernel 1, the blocks are 1D of size 32x32 instead of (32, 32)
  // I think this kernel only works if the number of threads in a block is the square of BLOCKSIZE.

  assert(blockDim.x == BLOCKSIZE**2);
  const uint x = blockIdx.x * BLOCKSIZE + threadIdx.x / BLOCKSIZE;
  const uint y = blockIdx.y * BLOCKSIZE + threadIdx.x % BLOCKSIZE;

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
```

* Compared to kernel 1, the shape of our blocks is different: now it is a 1D 32x32 block rather than a 2D (32, 32) block.
* The template parameter `BLOCKSIZE` is 32 and can be thought of as the edge size of the square grid of threads for our mapping from threads to entries of `C`.
* What happens if the kernel is called with a block shape that is not `BLOCKSIZE**2`? Bad things, I think we're making pretty specific assumptions about how to map threads to entries of `C`. I added an assert to protect against it.

> EXERCISE: What if we keep our block shape from Kernel 1 and just swap the coordinate assignments for `x` and `y`? Answer: the 2048x2048x2048 gemm runs at ~900 GFLOPs/s but the 4096x4096x4906 gemm runs at 648.0 GFLOPs/s. Weird.
> FOLLOW-UP: Why is that? Profile with nSight?

* kernel 2 (global memory coalescing): 813.3 GFLOPs

We're mainly using global memory and registers so far. GPU blocks also have shared memory (SMEM), which are shared between all threads in the same block. We can improve performance more by loading chunks of A and B into SMEM and doing as much work as possible. For the next kernel, each thread will still be assigned to a single entry of C.

  ###### kernel 3 implementation notes:
* Kernel 3 introduces the concept of thread synchronization / blocking for the first time. When e.g. loading matrix entries into SRAM, we want to make sure that loading is finished before we start computation.
* I ran into accuracy issues for the first time with this kernel. A couple ways we can debug:
	* we can add print statements to code with e.g. `printf("Thread %d: load_A = %u\n", threadIdx.x, load_A);` but it was a little overwhelming to see so many threads printing stuff out.
	* we can also run gdb with `DEVICE=0 cuda-gdb --arg ./sgemm 3`. One can switch threads and print out indices. ChatGPT says we need to rebuild in debug mode though.
* For some reason, my implementation of the original Kernel 2 and 3 reach their peak performance at 2048 x 2048 x 2048 compared to 4096 x 4096 x 4096, though it's less dramatic than my variant of Kernel 2. E.g. Kernel 3 gets 1226.5 GFLOPs with 2048 but 1127.7 GFLOPs with 4096. The same holds when I rebuild and run Simon's version of Kernel 3, but not Kernel 2 for some reason.
	* What's the difference between my Kernel 2 and Simon's original Kernel 2? Comparing the code, I see a couple differences:
		* Simon defines float tmp = 0.0, I just define it as float tmp = 0;
		* I iterate over an int64_t rather than an int
		* Simon did ++i instead of i++
		* I defined my index variables as const uints instead of const ints.
	* Turns out the last one was the problem. When I change them to const ints like Simon's original implementation, performance improves to 932.3 GFLOPs. I'm guessing this is a weird compiler thing but it'd be interesting to understand what's happening differently. 
```
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
```
kernel 3 (shared memory blocking): 1108.6 GFLOPs

Let's calculate Kernel 3's occupancy on my GPU. First we can collate some numbers on the GPU and the kernel:
```
From Nvidia Nsight + a nice helper function Simon already had implemented:

* Name: RTX 4060
* Compute Capability: 8.9
* max threads per block: 1024
* max threads per multiprocessor: 1536
* threads per warp: 32
* warp allocation granularity: 4
* max regs per block: 65536
* max regs per multiprocessor: 65536
* reg allocation unit size: 256
* reg allocation granularity: warp
* total global memory: 7940MB
* max shared mem per block: 100KB
* CUDA runtime shared mem overhead per block: 1024
* shared mem per multiprocessor: 102400
* multiprocessor count: 24
* max warps per multiprocessor: 48

  

My kernel's numbers:
* Registers per thread: 37
* SMEM per block: 8192 bytes
* Threads per block: 1024
```
**Occupancy calculation**

Occupancy is defined as the ratio between the number of active warps per SM and the maximum number of active warps per SM. In my case, the maximum number of warps per SM is 48.

There are three main things that can limit occupancy: register count, warp count, and SMEM capacity. Warps are scheduled on a block granularity, aka each SM will load as many blocks as it can accommodate.

* register count: 37 registers per thread * 32 threads per warp = 1184 registers per warp. Since we allocate registers in blocks of size 256, we need to allocate 1280 registers per warp. We have (1024 threads per block) / (32 threads per warp) = 32 warps per block, which means 40960 registers per block. The limit is 65536 registers per block, so we can fit one block per SM by this measure.
* warp count: 1024 threads / (32 threads per warp) => 32 warps. We can handle 32 warps per block as mentioned above, so we can fit one block per SM again by this measure.
* SMEM capacity: 8192 bytes per kernel + 1024 bytes of runtime overhead => 9216 bytes per block. (102400 SMEM bytes per MP) / (9216 SMEM bytes per block) => max 11 blocks per SM by this measure.

tl;dr: we're also limited by register and warp count to one block per SM, so 32 warps per SM. Each SM can handle at most 48, so like in Simon's case we're also getting 32/48 = 67% occupancy, pretty good.

We can profile Kernel 3 with Nvidia NSight with `make profile KERNEL=3`.  I took a couple screenshots -- they look pretty similar to Simon's example, and show that 1) loads are our most common instruction and 2) we spend a lot of WARP state cycles stalling due to IO pressure. 

> ASIDE: as mentioned above, the 2048 x 2048 x 2048 gemm actually achieved higher FLOPs utilization than the 4096 x 4096 x 4096 gemm. Staring at the warp state comparison in the NSight profile, it seems like the 4096 x 4096 x 4096 gemm spends more time in the "Stall Long Scoreboard" section. In the "GPU Speed of Light Throughput" section, it looks like we're using a lot more DRAM bandwidth. In the "Memory Workload Analysis" section, I can see that the L2 cache hit rate is ~100% for 2K but ~50% for 4K, and the total traffic from device memory to the L2 cache went up by 17x even though the size of the gemm went up by 7x. I guess what happened is that the L2 cache was filling up more quickly for the bigger gemm, and we started to be bottlenecked by this DRAM -> L2 cache bandwidth.
> Follow-up: The L2 cache on my GPU is 24MB. Figure out more details. Question for Leon rereading this later: will speeding up the kernel even help given that I'm currently bottlenecked by DRAM bandwidth? Why if so?

Kernel 4:

Basically, change our tensor block shape so that the A chunks are tall and skinny, and the B chunks are long and short, so that after we load into SMEM we can compute a bigger chunk of C. Now each thread computes multiple entries of `C`, which helps increase arithmetic intensity. 

Kernel 4 implementation notes:
* We load a `BM x BK` chunk of `A` into `As` and a `BK x BN` chunk of `B` into `Bs`. This lets us calculate a `BM x BN` chunk of the output gemm using SMEM. We will divvy this up so that each thread calculates a `TM x 1` chunk of the output. That means we will need `BM x BN / TM` threads.
* My initial attempt produced my first illegal memory access error! After building with debug mode, we can run the kernel with compute-sanitizer and get a stack trace with line numbers; it turns out I was treating the chunk of B we're loading into SMEM as the same shape as A, instead of transposed.
* After that, and a bit more debugging, the first 8 entries were right but I was getting wrong entries starting with entry 8. Actually the entire first 8 columns seem to match but I was getting a deviation starting with column 8. Turns out the reason is that I was doing const int x = TM * threadIdx.x / BN; which means when TM = 8 and BN = 64 that we accidentally were switching rows once threadIdx.x = 8. Oops!
* The initial correct version of Kernel 4 got 2511.4 GFLOPs, but I noticed that Simon's version achieved 3491.0 GFLOPs. Looking at the top of Simon's version, I see a comment about switching `blockIdx.x` and `blockIdx.y` to improve L2 cache hit rate. When I make that change Kernel 4 gets 3403.0 GLOPs, close enough.
```
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
```
Kernel 4 (with swapped blockIdxs): 3403.0 GFLOPs

Let's calculate how many GMEM loads from A -> As we had for Kernels 3 and 4 on the 4096 x 4096 x 4096 gemm.
* Kernel 3: For each block, we loaded 1024 entries 4096/32=128 times. There were 128 * 128 blocks, so in total we loaded 2.147e9 entries from A.
* Kernel 4: For each block, we loaded 512 entries 4096/8=512 times. There were 64 * 64 blocks, so in total we loaded 1.074e9 entries from A.
The same goes for B; we also load and store 1.678e7 entries for C. In other words, the arithmetic intensity roughly doubled :-)

> We can see in the NSight profile that the largest contributor to Warp Stall for the 4096 x 4096 x 4096 gemm was Stall Not Selected, which apparently means we might have too many active warps. I wonder how you improve this. 
> 
> For the 2048 x 2048 x 2048 gemm (which ran at 3555.3 GFLOPs) the largest stall contributor was Stall MIO Throttle, which basically means we're using too much memory bandwidth still.

So let's up the arithmetic intensity even more -- instead of each thread computing a small column of C, we'll aim to calculate a TMxTN subchunk of C.

###### Kernel 5 implementation notes:
* We load a `BM x BK` chunk of `A` into `As` and a `BK x BN` chunk of `B` into `Bs`. This lets us calculate a `BM x BN` chunk of the output gemm using SMEM. We will divvy this up so that each thread calculates a `TM x TN` chunk of the output. That means we will need `(BM x BN) / (TM x TN)` threads. For most gemm shapes, we will have `BM = 128 = BN` and `BK = TM = TN = 8`. So we'll have 256 threads per block, and when filling `As` and `Bs` each thread will need to fill in `128 x 8 / 256 = 4` entries.
* To make things even faster, when each thread switches from populating `As` and `Bs` to computing the `TM x TN` output, we load the relevant parts of `As` and `Bs` from SMEM to registers for even more data locality.
* We likely wanna be careful to make sure that the memory accesses are coalesced for both `As` and `Bs`, and use the same trick of using the less-intuitive choice of `blockIdx` to improve L2 cache hit rates.
* The initial correct version of my code ran at 1532.5 GFLOPs, much slower than Kernel 4. 
	* My first issue is that I `TM x BK` and `BK x TN` registers instead of just looping over BK. [Making BK the outermost loop](https://github.com/leonyz/SGEMM_CUDA/commit/7c16b7eab54e4e8c66f411019402754f0a77168a) brought my kernel to 4622.7 GFLOPs; faster than Kernel 4 but still slower than Simon's implementation which gets 6877.9 GFLOPs. 
	* Rereading the blogpost more carefully, I noticed that I was loading into As and Bs by having each thread handle consecutive entries of A and B rather than doing strided memory accesses. [Fixing this](https://github.com/leonyz/SGEMM_CUDA/commit/8c84538141f0d897888c415ccbfd94aee3fc47e1) took me to 5227.5 GFLOPs.
	* More experimentation revealed that the remaining big difference was that my kernel calculated some indices using `blockDim.x` while the original kernel calculated an equivalent value as `BM * BN / TM / TN`. [o3 says](https://chatgpt.com/share/682254b5-1878-8001-948f-a0b5eeb4d237) this is because `blockDim.x` is fetched from constant memory and is copied into a register of every thread in the warp, which means only one warp at a time can issue the load and compiler can't take advantage of the optimizer. (Godbolt [here](https://godbolt.org/#z:OYLghAFBqd5TKALEBjA9gEwKYFFMCWALugE4A0BIEAZgQDbYB2AhgLbYgDkAjF%2BTXRMiAZVQtGIHgBYBQogFUAztgAKAD24AGfgCsp5eiyahUAV0wtyKxqiIEh1ZpgDC6embZMQAJnLOAGQImbAA5TwAjbFJfWQAHdCViByY3Dy9fcgSk%2ByEgkPC2KJifWRtsOxSRIhZSIjTPbz9yyqFq2qJ8sMjo2OsauoaM5oHO4O6i3tKASmt0M1JUTi4AUh8AZjjSFmA2FgBqISWVrQBBE/ON4NQPHH2V9ZdxJRU6h9wLtfXr2%2Bx7x9QSiIhHQ70%2BVyYNwsfweTyBmHoBAiYLOXx%2B0P%2BTzMESMSgA%2BgA3HwAOiQKMu30hv0x5kseNIZmEBA4pPJFzxeOA9HQEQkHP2BPQBEw%2ByUwGwbDYnO5vPoeI4UowEmwSiWEGCRH2AFlyPsNftQrr9QBpXU0bksTUSOJIKwXfYOx1O50u11ul0YJhA/bm9CW/YAKlOus93t9/oDACFyPb3XH4/Hw5qojUzRbNQGXNN7gB2SOx/WRgIAeRcxpEAEkAFofdYAEXWPge%2BbODtDmv1qAASugAO7/Ov7HHoVAAawrmHUxPUgf2RdL5eruHuPkj%2BwgRCQpGwLEwE6nM4A9HOS2XKzXps3Y%2B29cJ9qg0gOh9yx/viQBPWfzs9Lldrjdbjue6TtOK4AKwngu564Je6wtucrb7Iex4EDQoo1EQErMB2Sj7CESwvLUn4kPseyjn8m7BMAuG9mQo77IyOCkPs9iMPsACOZjGPYABelopAWaEQN2faYtqK4AGxrBJ96PrCBrZiseaxg6SYsWwcRPloxJaFeiGqWQ676gQWnNnqYnGmZaz5quBCKcp%2BmOkQGl/g8g7nGBkYif2Ab7Maf56isYGDr5%2BaeSZvmhAFD7uEFdZ6acTpKfFjkuEFXk9j5BrRWkcVPtatqzs5mnWUO2A1LOaWed5s5RaVMX0HFCUOslnw5nWXCzPQ3Bgfw3hcDo5DoNwLgKHWiVpaupVKPMiwwhsfDkEQ2idbMSBAb0ECzKOsTEgAnAdh1HUdEmGNw0h9StQ3cPwSggFoS0rbMcCwCgGAaQw0SUNQ71xJ9MRMASqCoDwPA5uQOAEgQSwAGoENgvbFnEzBXXQ9CYaQd0QBEV0RMERHcIt70cMIxZMPQ75XTgewmJIA38IQ26VASKpXdg6gVGYmGE/wGrYN19OGEi2ykO%2Bbg4FdRCkMyPOzOaOxKHDCNIyjvD8IIwhiMqUiyBrigqBoV36DwhjGKYFhWIiER3ZAszoHEuRetwAC0xbrPszt1qEda4JGCgAOIe874ohNsmHOxgOBubUqBkvWv1c9geIABynZH2BuelCeYSnp1KFKqd5c7zsO%2BoLBKM79uO0obnOwSHvwm5qDWdZYO3fzFSO04TCuO4jQGIE4yFMUBjZMkQhDN4Jtj47XTD70JstI77SDH3GSLx3rRMCvYwFD0MSL6Mk8GECHRz/vUizDNCxLL4XU9ZdgvDVwBowy4Lj7KDxI5jp674MQhkvg%2BGmPwZa9NphrQ2jELa5AdqlH2sdRBB1ToCwuuQfqg1n63Xuo9cB5AXrIDQFgPAhASAUCoLQT6rAOA8zkJrcQkgZB0P1moTQgt9B%2BCMCYNAFtrCby7hAZwx8/CDz3pMEo8REjj1SGvJoWQpGzyHhfUofDbDLyPrIzIS8qijHPuIvop9V7pDkYY3eEwR4zDmDfZYXwtg7D2IcSEGdUSoghFCO4sJnivCIGyNx1JPHwgcL4yk7iYQAnhFbYJ6IPEAmxLiQkJI44fFcSE/xsS6QMiZCyJJnwzgci5DyPkeIBRChFGKCUUoCmynlBKPEjIlSMFVNgTA6o7w6lvJqQ0HS/Jpj9FaegNo7SOQTCMkZN41JBhDEIMM6ZAzRhUqMxZcY1IpisD6WZmZ7LwQdIWU8i4axuUbM1e80yOx3hqm5Z8I5xwgRnKFPZ0EAoAW3LuN8M41gQW/PsmCxybydjkvWK5r4QKfnuVBX8pVnlATeUhSCP4LwJVjMhPUaFT6YRJjhPCFQVRKCIixdApEWDkRYkgKiNE6IMR7tEFin0OJcSZHxR2gl1wXMeOJaS0lZLuDEqELZCy1LFS0jpY5BlmKtI7KZOC5l5KWSlS3WyfLhnqRKlNQFHkMqiV8v5UqBA8qhXShFbK9VcrBRFbmFKiVHRVQ1VlOqq4uWNWCvlAZhVfKCtKqsyqWdMq1RyrFU1cFYytVRO1TqZ0uC9XQVdZ%2Bo1xr7EmjZNc185orkbKAp6kDdybW2rtJBSCUHnX4GwEA6wf5gSjU/G61gcFgJ0M9RAEA3roA%2BowchP1m1/VbSAQGwNQbg0htDbAStEbIwwerBgGMsY40FnjVgotaHE2wmTCmVMJRmzpoNRmncCAszuoLdmnNuZq0oMIfmV0rYizFsQyW0ti1qzlkYaiw6Vb9UWnrLWjDdbyGUKwo2mQuHm0sELa28A7YOxSHu127tPbe19gHIOIdoiWmwBHYh0dFhxzrNnJOqdyDp0ztVDtidc7WALlJJ1xdS7l0ruB6Ztd67O0bvHeVkY26qO3Y4QRPdj4m1EeYhe8icgpB44J6ReiR4bzUTojoIntFtF0Uo/Rh8ZOaOU3UcTC8r6zVvjwe%2BEbH6YO4K/d%2Bn8eDf1/hAf%2BZDU26fTeAzNTFqA5vgXmxBBauBoLHddLg2CHq1tWuGnwRaQBgW0snHge0JI5h8DwdYoNpA5jAuWrzWDcF1vwQ2htKAB1LG%2BsJDt/1QjsGWKEN%2BH8v4/0Gs0gB0tMAGHfQwzgTC9Y/sNuwrR/CUjd17sYgePcNMH1E47ETM8UgDZPp1%2BTKneuSY49vBTYiJP9Gm/3NTZj54H1mFLbA2BhT3T05GlL3A6zYChksfYw7qWlZMxVizVnAELV1G4Ft1KgEgLSwF9aWboE5ukBJYkElwtA54MDoH4a0HFp4FoB6R2fPVr809DL8BG0gBy9gPLv1CvFe4Nd8rZnKv8Gq2QvbfgGva2a9%2Bg2bDBrG3Y1vbrIm%2BMbdHgo4TqmhtjcU0tuT82Vvrzp%2Bos%2BXOBOmNkwt/jm2lrbl23VrQB2DP8Gfids7fxLvMVx6Z8zWg/6kIe42J7BXW02fe/5iB5AvuOZgTtMC6xiRxdByDx3ydOGFvIJD6HFbDNw7ugj%2BzgXgvSD2sSEHWhGw5lTsnZOYFSgSVOrDuzdac1Q4egLdYCvvMJ4CyzTGXXpBAA%3D%3D)). With that we reach 7140.9 GFLOPs: already about 78% of CuBLAS :-)

```
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

    // initialize As and Bs
    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];
    
    
    // calculate the upper-left corner of the chunk of A and B that our particular
    // thread is loading into As and Bs
    const uint num_threads = BM * BN / (TM * TN);

    const uint As_rows_per_load = num_threads / BK;
    const uint thread_As_entries_to_load = BM * BK / num_threads;
    const uint thread_As_load_row = threadIdx.x / BK;
    const uint thread_As_load_col = threadIdx.x % BK;

    const uint Bs_rows_per_load = num_threads / BN;
    const uint thread_Bs_entries_to_load = BK * BN / num_threads;
    const uint thread_Bs_load_row = threadIdx.x / BN;
    const uint thread_Bs_load_col = threadIdx.x % BN;

    // initialize the parts of As and Bs we'll use for computing entries of C
    float A_local[TM] = {0.0};
    float B_local[TN] = {0.0};

    // initialize the running reductions for C
    float C_out[TM*TN] = {0.0};

    if ((A_row + thread_row < M) && (B_col + thread_col < N)) {
      for (int i = 0; i * BK < K; i++) {
        // step 0: load from A and B into As and Bs
        for (int load_idx = 0; load_idx < thread_As_entries_to_load; load_idx++) {
          As[thread_As_load_row * BK + thread_As_load_col + load_idx * As_rows_per_load * BK] = A[thread_As_load_row * K + thread_As_load_col + load_idx * As_rows_per_load * K];
        }
        for (int load_idx = 0; load_idx < thread_Bs_entries_to_load; load_idx++) {
          Bs[thread_Bs_load_row * BN + thread_Bs_load_col + load_idx * Bs_rows_per_load * BN] = B[thread_Bs_load_row * N + thread_Bs_load_col + load_idx * Bs_rows_per_load * N];
        }

        __syncthreads();

        // step 1: load from As and Bs into A_local and B_local
        for (int BK_idx = 0; BK_idx < BK; BK_idx++) {
          for (int A_local_row = 0; A_local_row < TM; A_local_row++) {
            A_local[A_local_row] = As[thread_row * BK + A_local_row * BK + BK_idx];
          }
          for (int B_local_col = 0; B_local_col < TN; B_local_col++) {
            B_local[B_local_col] = Bs[thread_col + BK_idx * BN + B_local_col];
          }
        // step 2: compute C_out reduction for this blocktile
          for (int C_out_row = 0; C_out_row < TM; C_out_row++) {
            for (int C_out_col = 0; C_out_col < TN; C_out_col++) {
              C_out[C_out_row * TN + C_out_col] += A_local[C_out_row] * B_local[C_out_col];
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
```
---------------------------
# Original README!

# Fast CUDA SGEMM from Scratch

Step-by-step optimization of matrix multiplication, implemented in CUDA.
For an explanation of each kernel, see [siboehm.com/CUDA-MMM](https://siboehm.com/articles/22/CUDA-MMM).

## Overview

Running the kernels on a NVIDIA A6000 (Ampere):

![](benchmark_results.png)

GFLOPs at matrix size 4096x4096:
<!-- benchmark_results -->
| Kernel                              |  GFLOPs/s | Performance relative to cuBLAS |
|:------------------------------------|----------:|:-------------------------------|
| 1: Naive                            |   `309.0` | 1.3%                           |
| 2: GMEM Coalescing                  |  `1986.5` | 8.5%                           |
| 3: SMEM Caching                     |  `2980.3` | 12.8%                          |
| 4: 1D Blocktiling                   |  `8474.7` | 36.5%                          |
| 5: 2D Blocktiling                   | `15971.7` | 68.7%                          |
| 7: Avoid Bank Conflicts (Linearize) | `16213.4` | 69.7%                          |
| 8: Avoid Bank Conflicts (Offset)    | `16459.2` | 70.8%                          |
| 11: Double Buffering                | `17278.3` | 74.3%                          |
| 6: Vectorized Mem Access            | `18237.3` | 78.4%                          |
| 9: Autotuning                       | `19721.0` | 84.8%                          |
| 10: Warptiling                      | `21779.3` | 93.7%                          |
| 0: cuBLAS                           | `23249.6` | 100.0%                         |
<!-- benchmark_results -->

## Setup

1. Install dependencies: CUDA toolkit 12, Python (+ Seaborn), CMake, Ninja. See [environment.yml](environment.yml).
1. Configure NVCC compilation parameters. Look up your GPUs compute
   capability [here](https://developer.nvidia.com/cuda-gpus). Then configure the `CMakeLists.txt` and change:
    ```cmake
    set(CUDA_COMPUTE_CAPABILITY 80)
    ```
1. Build: `mkdir build && cd build && cmake .. && cmake --build .`
1. Run one of the kernels: `DEVICE=<device_id> ./sgemm <kernel number>`
1. Profiling via [NVIDIA Nsight Compute](https://developer.nvidia.com/nsight-compute) (ncu): `make profile KERNEL=<kernel number>`

Credit goes to [wangzyon/NVIDIA_SGEMM_PRACTICE](https://github.com/wangzyon/NVIDIA_SGEMM_PRACTICE) for the benchmarking setup.
