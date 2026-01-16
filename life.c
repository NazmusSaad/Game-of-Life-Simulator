/*****************************************************************************
 * life.c
 * Parallelized and optimized implementation of the game of life resides here
 ****************************************************************************/
#include "life.h"
#include "util.h"
#include <pthread.h>
#include <immintrin.h>

/*****************************************************************************
 * Helper function definitions
 ****************************************************************************/
#define SWAP_BOARDS( b1, b2 )  do { \
	char* temp = b1; \
	b1 = b2; \
	b2 = temp; \
} while(0)

#define BOARD( __board, __i, __j )  (__board[(__i) + LDA*(__j)])

static inline __m256i alivep_avx2(__m256i count, __m256i state){
	// this function returns the result of (sum==3 | (state & (sum==2)))

	// set the constant vectors
	const __m256i twos = _mm256_set1_epi8(2);
	const __m256i threes = _mm256_set1_epi8(3);
	const __m256i ones = _mm256_set1_epi8(1);

	// check the count
	__m256i equal_to_2 = _mm256_cmpeq_epi8(count, twos);
	__m256i equal_to_3 = _mm256_cmpeq_epi8(count, threes);

	// check if count == 3
	__m256i born = _mm256_and_si256(equal_to_3, ones);

	// check if it survives by state & count==2
	__m256i survives = _mm256_and_si256(equal_to_2, state);

	// result = born | survives
	__m256i result = _mm256_or_si256(born, survives);
	return result;

}

// thread info strcut that gets passed into worker
typedef struct {
	// char *inboard;
	// char *outboard;
	int nrows, ncols;
	int start, end;
	int num_generations;
	int tid;
	pthread_barrier_t *barrier;
	// shared ptrs among threads to help swap in and out boards
	char **inptr, **outptr;
} worker_arg;

static void* worker(void* a){
	worker_arg* wa = (worker_arg*) a;

	// update local inboard and outboard ptrs for each thread.
	char *inboard = *(wa->inptr);
	char *outboard = *(wa->outptr);

	// wrap with bitmasks since % is slow
	int row_mask = wa->nrows - 1;
	int col_mask = wa->ncols - 1;
	
	for (int curgen = 0; curgen < wa->num_generations; ++curgen){
		// loop thru the portion of inboard for current thread and compute the outboard rows in parallel
		for (int i = wa->start; i < wa->end; ++i){
			int col_window = 2048;

			int ncols = wa->ncols;

			const int inorth = (i-1) & row_mask;
			const int isouth = (i+1) & row_mask;
			// save multiplication by calculating here
			int rn = inorth * ncols;
			int r0 = i * ncols;
			int rs = isouth * ncols;

			// do tiling so that a window of 3 rows by col_window columns stays hot in the cache
			for (int j_outer = 0; j_outer < wa->ncols; j_outer += col_window){
				// // jend will be some multiple of col_window or the end of the row
				// int jend = j_outer + (j_outer + col_window<wa->ncols? col_window:wa->ncols-j_outer);

				const int jend = ncols - 1;

				// process j = 0 using scalar method
				int j = 0;
				const int jwest = (j-1) & col_mask;
				const int jeast = (j+1) & col_mask;
					

				// replaced the col major with row major and replaced the unnecsary use of the macro 
				const char neighbor_count =
					inboard[rn + jwest] +
					inboard[rn + j] +
					inboard[rn + jeast] +
					inboard[r0 + jwest] +
					inboard[r0 + jeast] +
					inboard[rs + jwest] +
					inboard[rs + j] +
					inboard[rs + jeast];

				outboard[r0 + j] = alivep(neighbor_count, inboard[r0 + j]);
				j = 1;
				// process multiples of 32 in the range [1, ncols - 2] by treating them as vectors
				for (; j + 31 < jend; j+=32) {

					// load in the 8 cardinal directions and state
					__m256i NW = _mm256_loadu_si256((const __m256i*)(inboard + rn + (j-1)));
					__m256i N = _mm256_loadu_si256((const __m256i*)(inboard + rn + (j)));
					__m256i NE = _mm256_loadu_si256((const __m256i*)(inboard + rn + (j+1)));

					__m256i W = _mm256_loadu_si256((const __m256i*)(inboard + r0 + (j-1)));
					__m256i state = _mm256_loadu_si256((const __m256i*)(inboard + r0 + (j)));
					__m256i E = _mm256_loadu_si256((const __m256i*)(inboard + r0 + (j+1)));

					__m256i SW = _mm256_loadu_si256((const __m256i*)(inboard + rs + (j-1)));
					__m256i S = _mm256_loadu_si256((const __m256i*)(inboard + rs + (j)));
					__m256i SE = _mm256_loadu_si256((const __m256i*)(inboard + rs + (j+1)));

					// now count the number of neighbours from the 8 directions
					__m256i neighbor_count = _mm256_add_epi8(NW, N);
					neighbor_count = _mm256_add_epi8(neighbor_count, NE);
					neighbor_count = _mm256_add_epi8(neighbor_count, W);
					neighbor_count = _mm256_add_epi8(neighbor_count, E);
					neighbor_count = _mm256_add_epi8(neighbor_count, SE);
					neighbor_count = _mm256_add_epi8(neighbor_count, SW);
					neighbor_count = _mm256_add_epi8(neighbor_count, S);

					// get the results from the function
					__m256i result = alivep_avx2(neighbor_count, state);
					
					// update the board
					_mm256_storeu_si256((__m256i*)(outboard + r0 + j), result);
				}

				// process the remaining rows (j, ncols) using the scalar method
				for (; j < ncols; ++j){
					const int jwest = (j-1) & col_mask;
					const int jeast = (j+1) & col_mask;
					

					// replaced the col major with row major and replaced the unnecsary use of the macro 
					const char neighbor_count =
						inboard[rn + jwest] +
						inboard[rn + j] +
						inboard[rn + jeast] +
						inboard[r0 + jwest] +
						inboard[r0 + jeast] +
						inboard[rs + jwest] +
						inboard[rs + j] +
						inboard[rs + jeast];
	
					outboard[r0 + j] = alivep(neighbor_count, inboard[r0 + j]);
				}
			}
		}

		// make sure every thread is finished computing their portion of output board
		pthread_barrier_wait(wa->barrier);

		// swap the boards by swapping the pointers to a pointer
		char** temp = wa->inptr;
		wa->inptr = wa->outptr;
		wa->outptr = temp;
		

		// refrsh the boards with the swap
		inboard = *(wa->inptr);
		outboard = *(wa->outptr);
		
	}

}


/*****************************************************************************
 * Game of life implementation
 ****************************************************************************/
char *
game_of_life(char *outboard, char *inboard, const int nrows, const int ncols,
	     const int num_generations)
{
	// decide on number of threads
	int num_threads = 16; // based on specs on piazza

	// initialize the stuff needed for threads
	pthread_t threads[num_threads];
	worker_arg worker_args[num_threads];
	pthread_barrier_t barrier;
	pthread_barrier_init(&barrier, NULL, num_threads);

	int base = nrows / num_threads;
	int remainder = nrows % num_threads;

	for (int t = 0; t < num_threads; ++t){
		// partition the rows in a way that the first remainder threads get 1 extra row after the initial division
		int start = base * t + (t < remainder ? t : remainder);
		int end = start + base + (t < remainder ? 1 : 0);

		worker_arg wa = {
			.barrier = &barrier,
			.inptr = &inboard,
			.outptr = &outboard,
			.start = start,
			.end = end,
			.ncols = ncols,
			.nrows = nrows,
			.num_generations = num_generations,
			.tid = t,
		};
		worker_args[t] = wa;

		pthread_create(&threads[t], NULL, worker, &worker_args[t]);
	}

	// join
	for (int t = 0; t < num_threads; ++t){
		pthread_join(threads[t], NULL);
	}
	pthread_barrier_destroy(&barrier);

	
	
	return inboard;
}
