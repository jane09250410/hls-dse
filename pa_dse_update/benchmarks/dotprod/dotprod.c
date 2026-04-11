/**
 * Dot Product (dotprod)
 * --------------------
 * Inner product of two vectors: result = sum( A[i] * B[i] )
 *
 * HLS-DSE characteristics:
 *   - Single loop with scalar accumulation (strong reduction dependency)
 *   - Stride-1 memory access on both arrays
 *   - Pipeline will conflict with the scalar accumulator
 *   - Simpler than matmul (single loop vs triple-nested)
 *   - Tests PA-DSE on the most basic accumulation pattern
 */

#define N 256

int dotprod(int A[N], int B[N]) {
    int i;
    int sum = 0;
    for (i = 0; i < N; i++) {
        sum += A[i] * B[i];
    }
    return sum;
}
