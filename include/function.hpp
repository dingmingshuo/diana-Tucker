#ifndef __DIANA_CORE_INCLUDE_FUNCTION_HPP__
#define __DIANA_CORE_INCLUDE_FUNCTION_HPP__

#include "tensor.hpp"

namespace Function {
    // Matrix functions
    template<typename Ty>
    Tensor<Ty> matmulNN(const Tensor<Ty> &A, const Tensor<Ty> &B);

    template<typename Ty>
    Tensor<Ty> matmulNT(const Tensor<Ty> &A, const Tensor<Ty> &B);

    template<typename Ty>
    Tensor<Ty> matmulTN(const Tensor<Ty> &A, const Tensor<Ty> &B);

    template<typename Ty>
    Tensor<Ty> inverse(const Tensor<Ty> &A);

    template<typename Ty>
    Tensor<Ty> transpose(const Tensor<Ty> &A);

    template<typename Ty>
    std::tuple<Tensor<Ty>, Tensor<Ty>> reduced_LQ(const Tensor<Ty> &A);

    template<typename Ty>
    std::tuple<Tensor<Ty>, Tensor<Ty>> reduced_QR(const Tensor<Ty> &A);

    template<typename Ty>
    Tensor<Ty> gram(const Tensor<Ty> &A);

    // Tensor functions

    template<typename Ty>
    Tensor<Ty> ttm(const Tensor<Ty> &A, const Tensor<Ty> &M, size_t n);

    template<typename Ty>
    Tensor<Ty>
    ttmc(const Tensor<Ty> &A, const std::vector<Tensor<Ty>> &M,
         const std::vector<size_t> &idx);

    template<typename Ty>
    Tensor<Ty> gram(const Tensor<Ty> &A, size_t n);


    template<typename Ty>
    Tensor<Ty> gather(const Tensor<Ty> &A);

    template<typename Ty>
    Tensor<Ty>
    scatter(const Tensor<Ty> &A, Distribution *distribution, int proc);

    template<typename Ty>
    double fnorm(const Tensor<Ty> &A);

    template<typename Ty>
    Ty sum(const Tensor<Ty> &A);
} // namespace Function

#include "function/matrix.tpp"
#include "function/tensor.tpp"

#endif