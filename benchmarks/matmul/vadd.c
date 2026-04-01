/**
 * Vector Addition (vadd)
 * ----------------------
 * Simple element-wise addition: C[i] = A[i] + B[i]
 *
 * HLS-DSE characteristics:
 *   - No loop-carried data dependency
 *   - Regular, stride-1 memory access pattern
 *   - Pipeline should always be feasible (no dependence conflict)
 *   - Unrolling freely applicable
 *   - Failure profile: mainly parameter-incompatibility (e.g. channel mismatch)
 *
 * Expected behavior in PA-DSE study:
 *   - Low structured infeasibility compared to matmul
 *   - PA-DSE benefit comes from filtering obviously invalid combos
 *   - Serves as "easy" baseline benchmark
 */

#define N 128

void vadd(int A[N], int B[N], int C[N]) {
    int i;
    for (i = 0; i < N; i++) {
        C[i] = A[i] + B[i];
    }
}
