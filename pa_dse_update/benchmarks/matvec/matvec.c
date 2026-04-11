/**
 * Matrix-Vector Multiply (matvec)
 * -------------------------------
 * y[i] = sum_j( A[i][j] * x[j] )
 *
 * HLS-DSE characteristics:
 *   - Double-nested loop (outer: rows, inner: dot product)
 *   - Inner loop has accumulator dependency (same as FIR inner loop)
 *   - Regular stride-1 access on x[], row-major access on A[]
 *   - Complexity between vadd (trivial) and matmul (triple nested)
 *   - Pipeline conflict expected on inner loop accumulator
 */

#define M 32
#define K 32

void matvec(int A[M][K], int x[K], int y[M]) {
    int i, j;
    for (i = 0; i < M; i++) {
        int acc = 0;
        for (j = 0; j < K; j++) {
            acc += A[i][j] * x[j];
        }
        y[i] = acc;
    }
}
