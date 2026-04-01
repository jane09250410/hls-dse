/**
 * FIR Filter (fir)
 * ----------------
 * Finite Impulse Response filter: y[i] = sum_j( coeff[j] * x[i-j] )
 *
 * HLS-DSE characteristics:
 *   - Inner loop has accumulator dependency (acc += ...)
 *   - Outer loop is independent across output samples
 *   - Regular, sliding-window memory access pattern
 *   - Pipeline on inner loop may conflict with accumulator dependence
 *   - Unrolling inner loop may hit resource limits at high factors
 *
 * Expected behavior in PA-DSE study:
 *   - Moderate structured infeasibility
 *   - Pipeline feasibility depends on which loop is pipelined
 *   - Accumulator dependence creates a distinct failure profile from matmul
 *   - Good "middle ground" between vadd (easy) and matmul (hard)
 */

#define N_SAMPLES 128
#define N_TAPS    16

void fir(int input[N_SAMPLES], int output[N_SAMPLES], int coeffs[N_TAPS]) {
    int i, j;
    for (i = 0; i < N_SAMPLES; i++) {
        int acc = 0;
        for (j = 0; j < N_TAPS; j++) {
            int idx = i - j;
            if (idx >= 0) {
                acc += coeffs[j] * input[idx];
            }
        }
        output[i] = acc;
    }
}
