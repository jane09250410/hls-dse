/**
 * ATAX - Matrix Transpose and Vector Multiply (atax)
 * ---------------------------------------------------
 * Computes y = A^T * (A * x)   (two-phase kernel from PolyBench)
 *
 * HLS-DSE characteristics:
 *   - Two separate double-nested loops (phases)
 *   - Both phases have inner-loop accumulation
 *   - Different access patterns: row-major in phase 1, column-major in phase 2
 *   - Column-major access in phase 2 may stress memory subsystem
 *   - Tests PA-DSE with multi-phase kernels and mixed access patterns
 */

#define M 32
#define N 32

void atax(int A[M][N], int x[N], int y[N], int tmp[M]) {
    int i, j;

    /* Phase 1: tmp = A * x */
    for (i = 0; i < M; i++) {
        int acc = 0;
        for (j = 0; j < N; j++) {
            acc += A[i][j] * x[j];
        }
        tmp[i] = acc;
    }

    /* Phase 2: y = A^T * tmp */
    for (j = 0; j < N; j++) {
        int acc = 0;
        for (i = 0; i < M; i++) {
            acc += A[i][j] * tmp[i];  /* column-major access on A */
        }
        y[j] = acc;
    }
}
