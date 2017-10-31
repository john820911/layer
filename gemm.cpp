#include <cstdio>
#include <cstdlib>
#include "gemm.h"

void gemm_nn(
    const int M, const int N, const int K,
    const float *A, const int length_A,
    const float *B, const int length_B,
    float *NN, const int length_NN
) {
    for(int m = 0; m < M; ++m) {
        for(int k = 0; k < K; ++k) {
            for(int n = 0; n < N; ++n) {
                NN[m * length_NN + n] += A[m * length_A + k] * B[k * length_B + n];
            }
        }
    }
}

void gemm_nt(
    const int M, const int N, const int K,
    const float *A, const int length_A,
    const float *B_T, const int length_B_T,
    float *NT, const int length_NT
) {
    for(int m = 0; m < M; ++m) {
        for(int n = 0; n < N; ++n) {
            for(int k = 0; k < K; ++k) {
                NT[m * length_NT + n] += A[m * length_A + k] * B_T[n * length_B_T + k];
            }
        }
    }
}

void gemm_tn(
    const int M, const int N, const int K,
    const float *A_T, const int length_A_T,
    const float *B, const int length_B,
    float *TN, const int length_TN
) {
    for(int m = 0; m < M; ++m) {
        for(int k = 0; k < K; ++k) {
            for(int n = 0; n < N; ++n) {
                TN[m * length_TN + n] += A_T[k * length_A_T + m] * B[k * length_B + n];
            }
        }
    }
}

void gemm_tt(
    const int M, const int N, const int K,
    const float *A_T, const int length_A_T,
    const float *B_T, const int length_B_T,
    float *TT, const int length_TT
) {
    for(int m = 0; m < M; ++m) {
        for(int n = 0; n < N; ++n) {
            for(int k = 0; k < K; ++k) {
                TT[m * length_TT + n] += A_T[k * length_A_T + m] * B_T[n * length_B_T + k];
            }
        }
    }
}
