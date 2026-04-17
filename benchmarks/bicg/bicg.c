/**
 * BiCG: Sub Kernel of BiCGStab Linear Solver
 * Source: PolyBench/C 4.2.1
 */

#define N 32

void bicg(int A[N][N], int s[N], int q[N], int p[N], int r[N]) {
    int i, j;
    for (i = 0; i < N; i++) {
        s[i] = 0;
    }
    for (i = 0; i < N; i++) {
        q[i] = 0;
        for (j = 0; j < N; j++) {
            s[j] = s[j] + r[i] * A[i][j];
            q[i] = q[i] + A[i][j] * p[j];
        }
    }
}
