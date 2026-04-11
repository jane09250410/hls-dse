/**
 * 1D Stencil (stencil)
 * --------------------
 * 3-point stencil: out[i] = w0*in[i-1] + w1*in[i] + w2*in[i+1]
 *
 * HLS-DSE characteristics:
 *   - Single loop with neighbor memory accesses (window pattern)
 *   - Accumulation within each iteration (but no loop-carried dependency
 *     on the accumulator since each out[i] is independent)
 *   - Overlapping read pattern (in[i] read by 3 iterations)
 *   - Pipeline should be feasible (no true loop-carried dep on accumulator)
 *   - Tests whether PA-DSE correctly avoids false suppression
 */

#define N 256

void stencil(int in[N], int out[N], int w0, int w1, int w2) {
    int i;
    for (i = 1; i < N - 1; i++) {
        out[i] = w0 * in[i - 1] + w1 * in[i] + w2 * in[i + 1];
    }
    /* boundary */
    out[0]     = w1 * in[0] + w2 * in[1];
    out[N - 1] = w0 * in[N - 2] + w1 * in[N - 1];
}
