#ifndef __DIANA_CORE_INCLUDE_FUNCTION_HPP__
#define __DIANA_CORE_INCLUDE_FUNCTION_HPP__

#include "tensor.hpp"

namespace Function {
    template<typename Ty>
    Tensor<Ty> ttm(const Tensor<Ty> &A, const Tensor<Ty> &M, size_t n);

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

#include "function.tpp"

#endif