#include "operator.hpp"
#include "util.hpp"
#include "logger.hpp"
#include "tensor.hpp"

#include <cstdlib>
#include <cmath>

#ifdef DIANA_MKL
extern "C" {
#include "mkl_cblas.h"
#include "mkl_lapacke.h"
}
#else
extern "C" {
#include "cblas.h"
#include "lapacke.h"
}
#endif

template <>
void Operator<double>::add(double *C, double *A, double *B, size_t n) {
    for (size_t i = 0; i < n; i++) {
        C[i] = A[i] + B[i];
    }
}

template <>
void Operator<double>::sub(double *C, double *A, double *B, size_t n) {
    for (size_t i = 0; i < n; i++) {
        C[i] = A[i] - B[i];
    }
}

template <>
void Operator<double>::mul(double *C, double *A, double *B, size_t n) {
    for (size_t i = 0; i < n; i++) {
        C[i] = A[i] * B[i];
    }
}

template <>
void Operator<double>::nmul(double *C, double *A, double B, size_t n) {
    for (size_t i = 0; i < n; i++) {
        C[i] = A[i] * B;
    }
}

template <> void Operator<double>::constant(double *A, double c, size_t n) {
    for (size_t i = 0; i < n; i++) {
        A[i] = c;
    }
}

template <> void Operator<double>::randn(double *A, size_t n) {
    for (size_t i = 0; i < n; i++) {
        A[i] = Util::randn();
    }
}

template <> double Operator<double>::fnorm(double *A, size_t n) {
    double ret = 0;
    for (size_t i = 0; i < n; i++) {
        ret += A[i] * A[i];
    }
    return std::sqrt(ret);
}

template <>
void Operator<double>::matmulNN(double *C, double *A, double *B, size_t m,
                                size_t n, size_t k) {
    double alpha = 1.0;
    int lda = (int)m;
    int ldb = (int)k;
    double beta = 0.0;
    int ldc = (int)m;
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, (int)m, (int)n,
                (int)k, alpha, A, lda, B, ldb, beta, C, ldc);
}

template <>
void Operator<double>::matmulNT(double *C, double *A, double *B, size_t m,
                                size_t n, size_t k) {
    double alpha = 1.0;
    int lda = (int)m;
    int ldb = (int)n;
    double beta = 0.0;
    int ldc = (int)m;
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, (int)m, (int)n, (int)k,
                alpha, A, lda, B, ldb, beta, C, ldc);
}