#define N 32

void atax(int A[N][N], int x[N], int y[N], int tmp[N]) {
    int i, j;
    for (i = 0; i < N; i++) y[i] = 0;
    for (i = 0; i < N; i++) {
        tmp[i] = 0;
        for (j = 0; j < N; j++) tmp[i] = tmp[i] + A[i][j] * x[j];
        for (j = 0; j < N; j++) y[j] = y[j] + A[i][j] * tmp[i];
    }
}
