#ifndef __DIANA_CORE_INCLUDE_ALGORITHM_HPP__
#define __DIANA_CORE_INCLUDE_ALGORITHM_HPP__

#include "tensor.hpp"
#include "def.hpp"
#include <vector>

namespace Algorithm {
namespace GRQI {
template <typename T>
std::tuple<std::vector<Tensor<T>>, double>
decompose(const Tensor<T> &, const std::vector<Tensor<T>> &, double);
}; // namespace GRQI
}; // namespace Algorithm

#include "algorithm/grqi.tpp"

#endif