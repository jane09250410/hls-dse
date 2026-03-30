#include <stdio.h>
#include "matmul.h"

int main() {
    int A[N][N], B[N][N], C[N][N];
    int i, j;
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            A[i][j] = i + j;
            B[i][j] = i * j + 1;
        }
    }
    matmul(A, B, C);
    printf("Result matrix C:\n");
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            printf("%6d ", C[i][j]);
        }
        printf("\n");
    }
    return 0;
}
