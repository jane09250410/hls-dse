#define N 16

void gemm(int A[N][N], int B[N][N], int C[N][N], int alpha, int beta) {
    int i, j, k;
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            C[i][j] = C[i][j] * beta;
            for (k = 0; k < N; k++) {
                C[i][j] = C[i][j] + alpha * A[i][k] * B[k][j];
            }
        }
    }
}
