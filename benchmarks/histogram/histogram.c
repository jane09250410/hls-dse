/**
 * Histogram (histogram)
 * ---------------------
 * Count occurrences of values into bins: hist[data[i]] += 1
 *
 * HLS-DSE characteristics:
 *   - Data-dependent memory addressing (hist[data[i]])
 *   - Read-modify-write on histogram array creates true dependency
 *   - Irregular, non-stride memory access pattern
 *   - Pipeline may fail due to memory port conflicts on hist[]
 *   - Unrolling is constrained by memory port availability
 *
 * Expected behavior in PA-DSE study:
 *   - Distinct failure profile: memory-port conflicts, not loop-carried arithmetic
 *   - Pipeline failures may appear even when matmul-style dependence is absent
 *   - Demonstrates that PA-DSE handles different failure mechanisms
 *   - "Hard" benchmark with irregular behavior
 */

#define N_DATA 256
#define N_BINS 64

void histogram(int data[N_DATA], int hist[N_BINS]) {
    int i;

    /* Initialize bins */
    for (i = 0; i < N_BINS; i++) {
        hist[i] = 0;
    }

    /* Accumulate */
    for (i = 0; i < N_DATA; i++) {
        int bin = data[i] % N_BINS;
        if (bin < 0) bin += N_BINS;
        hist[bin] += 1;
    }
}
