/**
 * GEMM - General Matrix Multiply (gemm)
 * --------------------------------------
 * C = alpha * A * B + beta * C   (from PolyBench)
 *
 * HLS-DSE characteristics:
 *   - Triple-nested loop like matmul but with scaling constants
 *   - Inner loop has accumulation (C[i][j] += ...)
 *   - Additional multiply-add for beta*C scaling before inner loop
 *   - Slightly more complex than plain matmul
 *   - Tests whether PA-DSE handles additional arithmetic in loop body
 */

#define NI 16
#define NJ 16
#define NK 16

void gemm(int alpha, int beta, int C[NI][NJ], int A[NI][NK], int B[NK][NJ]) {
    int i, j, k;
    for (i = 0; i < NI; i++) {
        for (j = 0; j < NJ; j++) {
            C[i][j] = beta * C[i][j];
            for (k = 0; k < NK; k++) {
                C[i][j] += alpha * A[i][k] * B[k][j];
            }
        }
    }
}
