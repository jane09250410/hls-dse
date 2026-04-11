/**
 * Sparse Matrix-Vector Multiply (spmv)
 * -------------------------------------
 * y[i] = sum_j( val[j] * x[col[j]] )  for j in row_ptr[i]..row_ptr[i+1]
 * Simplified CSR-like SpMV with indirect indexing.
 *
 * HLS-DSE characteristics:
 *   - Outer loop over rows, inner loop variable-length
 *   - Indirect memory access via col[] array (data-dependent addressing)
 *   - Accumulation in inner loop (loop-carried dependency)
 *   - Combines challenges of histogram (indirect access) and fir (accumulation)
 *   - Branch in outer loop (variable-length inner loop)
 *   - Tests PA-DSE on irregular, data-dependent access patterns
 *
 * Note: Uses a simplified fixed structure for HLS compatibility.
 */

#define N_ROWS 32
#define NNZ    128  /* total non-zeros */

void spmv(int val[NNZ], int col[NNZ], int row_ptr[N_ROWS + 1],
           int x[N_ROWS], int y[N_ROWS]) {
    int i, j;
    for (i = 0; i < N_ROWS; i++) {
        int acc = 0;
        for (j = row_ptr[i]; j < row_ptr[i + 1]; j++) {
            acc += val[j] * x[col[j]];
        }
        y[i] = acc;
    }
}
