# Optimized Parallel Game of Life

High-performance implementation of **Conway’s Game of Life** in C, focused on **parallelism, SIMD vectorization, and cache-aware optimization**.  
This project parallelizes a baseline sequential simulator using **Pthreads** and accelerates the core update kernel with **AVX2 intrinsics**, achieving up to **240× speedup** over the naïve implementation.

---

## Overview

The Game of Life is a cellular automaton where each cell’s next state depends on the number of live neighbors. While conceptually simple, the update kernel is compute- and memory-intensive, making it a strong candidate for systems-level optimization.

This implementation focuses on:

- Thread-level parallelism (row striping with Pthreads)
- Data-level parallelism (AVX2 SIMD)
- Cache locality (tiling and row-major access)
- Synchronization correctness (barrier-based generation updates)

---

## Key Features

- Row-striped multithreading using 16 Pthreads  
- Barrier synchronization between generations  
- AVX2 vectorized update kernel  
- Cache-aware tiling to keep working sets hot  
- Toroidal wrap-around using bitmasking (no modulo)  
- Pointer-swapped double buffering (no data copying)

---

## Parallelization Strategy

### Thread Decomposition

- The board is divided **row-wise** across threads
- Rows are split as evenly as possible
- Each thread processes a contiguous block `[start, end)`

This minimizes false sharing and improves cache locality.


### Synchronization

- A `pthread_barrier` ensures all threads finish computing their portion of the next generation
- Only after all threads arrive at the barrier do we swap boards

This guarantees correctness while keeping synchronization overhead minimal.

---

## SIMD Vectorization (AVX2)

The core update kernel is vectorized using **AVX2 intrinsics**:

- Each vector processes **32 cells per iteration** (32 × 8-bit cells)
- Neighbor counts are computed using vector adds across the 8 surrounding directions
- Game of Life rules are applied branch-free using vector comparisons and bitwise operations

Scalar: 1 cell/iteration
AVX2: 32 cells/iteration

This is the **largest contributor** to the observed speedup.

---

## Cache Optimizations

Several optimizations target memory behavior:

- Row-major indexing to maximize spatial locality
- Tiling over columns (sliding window) so a 3×N region stays hot in cache
- Bitmask wrap-around instead of modulo to reduce instruction latency
- Double buffering with pointer swaps to avoid copying boards

---

## Board Update Flow

For each generation:

1. Threads compute assigned rows of the output board in parallel  
2. Barrier synchronization  
3. Input and output boards are swapped via pointer exchange  
4. Next generation begins  

inboard → compute → outboard
(barrier)
swap(inboard, outboard)


---


---

## Implementation Notes

- Cell states are stored as **bytes (`char`)** to maximize SIMD density
- AVX2 intrinsics are used explicitly (`immintrin.h`)
- Unaligned loads/stores (`_mm256_loadu_si256`) are used for simplicity
- The implementation assumes board dimensions are powers of two for efficient masking

---

## Performance

- **Up to 240× speedup** over the baseline sequential implementation
- Speedups come primarily from:
  - AVX2 vectorization
  - Multithreading
  - Cache-aware memory access

Exact performance depends on board size, CPU microarchitecture, and memory hierarchy.

---

## Build & Run

Example (adjust flags as needed):

```bash
gcc -O3 -march=native -mavx2 -pthread life.c -o life
```
