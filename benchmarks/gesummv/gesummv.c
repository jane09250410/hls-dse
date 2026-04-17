#define N 32

void gesummv(int A[N][N], int B[N][N], int x[N], int y[N],
             int tmp[N], int alpha, int beta) {
    int i, j;
    for (i = 0; i < N; i++) {
        tmp[i] = 0;
        y[i] = 0;
        for (j = 0; j < N; j++) {
            tmp[i] = tmp[i] + A[i][j] * x[j];
            y[i] = y[i] + B[i][j] * x[j];
        }
        y[i] = alpha * tmp[i] + beta * y[i];
    }
}
