#ifndef GEMM_H
#define GEMM_H 

void gemm_nn(
    const int M, const int N, const int K,
    const float *A, const int length_A,
    const float *B, const int length_B,
    float *NN, const int length_NN
);

void gemm_nt(
    const int M, const int N, const int K,
    const float *A, const int length_A,
    const float *B_T, const int length_B_T,
    float *NT, const int length_NT
);

void gemm_tn(
    const int M, const int N, const int K,
    const float *A_T, const int length_A_T,
    const float *B, const int length_B,
    float *TN, const int length_TN
);

void gemm_tt(
    const int M, const int N, const int K,
    const float *A_T, const int length_A_T,
    const float *B_T, const int length_B_T,
    float *TT, const int length_TT
);

#endif
