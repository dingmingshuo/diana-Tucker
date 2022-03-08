#include "tensor.hpp"
#include "logger.hpp"
#include <tuple>

namespace Algorithm {
    namespace Tucker {
        template<typename Ty>
        Tensor<Ty> ALS_(const Tensor<Ty> &A, const Tensor<Ty> &L_initial) {
            auto L = A.copy();
        }

        template<typename Ty>
        std::tuple<Tensor<Ty>, std::vector<Tensor<Ty>>>
        HOOI_ALS(const Tensor<Ty> &A, const shape_t &R,
                 const std::vector<Tensor<Ty>> &U_0) {
            const size_t kN = U_0.size();
            assert(R.size() == kN);
            // Initialize U.
            std::vector<Tensor<Ty>> U;
            for (const Tensor<Ty> &item: U_0) {
                U.push_back(item.copy());
            }
            // Start iteration.
            size_t k = 0;
            while (not convergent) {
                // Step ++.
                k = k + 1;
                for (size_t n = 0; n < kN; n++) {
                    auto Y = A.copy();
                    for (size_t i = 0; i < kN; i++) {
                        if (i == n) continue;
                        Y = Function::ttm(Y, U[i], i);
                    }
                    U[n] = Algorithm::Tucker::ALS_(Y, U[n]);
                }
            }
        }
    }
}