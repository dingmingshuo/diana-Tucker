#include "tensor.hpp"
#include "function.hpp"
#include "logger.hpp"
#include <tuple>

namespace Algorithm ::Tucker {
    template<typename Ty>
    Tensor<Ty> ALS_(const Tensor<Ty> &Y, size_t n, const Tensor<Ty> &L_initial,
                    size_t max_iter = 5) {
        auto local = Function::gather(Y);
        auto YYt = Function::gram<Ty>(Y, n);
        auto L = L_initial.copy();
        for (size_t iter = 0; iter < max_iter; iter++) {
            auto G = Function::matmulTN<Ty>(L, L);
            auto G_inv = Function::inverse<Ty>(G);
            auto LG_inv = Function::matmulNN<Ty>(L, G_inv);
            auto YYtLG_inv = Function::matmulNN<Ty>(YYt, LG_inv);
            auto G_R = Function::matmulTN<Ty>(LG_inv, YYtLG_inv);
            auto G_R_inv = Function::inverse<Ty>(G_R);
//            if (mpi_rank() == 0) {
//                auto I = Function::matmulNN(G_R, G_R_inv);
//                I.print();
//            }
            L = Function::matmulNN<Ty>(YYtLG_inv, G_R_inv);
        }
        auto[q, r] = Function::reduced_QR(L);
        return q;
    }

    template<typename Ty>
    std::tuple<Tensor<Ty>, std::vector<Tensor<Ty>>>
    HOOI_ALS(const Tensor<Ty> &A, const shape_t &R, size_t max_iter) {
        assert(R.size() == A.ndim());
        const size_t kN = A.ndim();
        const shape_t &I = A.shape_global();
        // Info
        output("Start Tucker::HOOI_ALS decomposition.. with max_iter = " +
               std::to_string(max_iter));
        // Initialize U.
        std::vector<Tensor<Ty>> U;
        for (size_t n = 0; n < kN; n++) {
            Tensor<double> U_rand({I[n], R[n]}, false);
            U_rand.randn();
            auto[q, r] = Function::reduced_QR < Ty > (U_rand);
            U.push_back(q);
        }
        // Start iteration.
        auto A_norm = (long double) Function::fnorm<Ty>(A);
        output("||A||_F = " + std::to_string(A_norm));
        size_t k = 0;
        for (size_t iter = 0; iter < max_iter; iter++) {
            output("Calculating iteration " + std::to_string(iter + 1) +
                   " ...");
            // Step ++.
            k = k + 1;
            for (size_t n = 0; n < kN; n++) {
                // TTMc
                auto Y = A.copy();
                for (size_t i = 0; i < kN; i++) {
                    if (i == n) continue;
                    auto Ut = Function::transpose<Ty>(U[i]);
                    Y = Function::ttm<Ty>(Y, Ut, i);
                }
                // ALS
                U[n] = Algorithm::Tucker::ALS_(Y, n, U[n]);
            }
            auto G = A.copy();
            for (size_t n = 0; n < kN; n++) {
                auto Ut = Function::transpose<Ty>(U[n]);
                G = Function::ttm<Ty>(G, Ut, n);
            }
            auto G_norm = (long double) Function::fnorm<Ty>(G);
            output("||G||_F = " + std::to_string(G_norm));
            output("Residual: sqrt(1 - ||G||_F^2 / ||A||_F^2) = " +
                   std::to_string(
                           sqrt(1 - (G_norm * G_norm) / (A_norm * A_norm))));
        }
        auto G = A.copy();
        for (size_t n = 0; n < kN; n++) {
            auto Ut = Function::transpose<Ty>(U[n]);
            G = Function::ttm<Ty>(G, Ut, n);
        }
        output("Done!");
        return std::make_tuple(G, U);
    }
}