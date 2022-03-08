#ifndef __DIANA_CORE_INCLUDE_ALGORITHM_HPP__
#define __DIANA_CORE_INCLUDE_ALGORITHM_HPP__

#include "tensor.hpp"
#include "def.hpp"
#include <vector>

namespace Algorithm {
    namespace Tucker {
        template<typename Ty>
        std::tuple<Tensor<Ty>, std::vector<Tensor<Ty>>>
        HOOI_ALS(const Tensor<Ty> &A, const shape_t R,
                 const std::vector<Tensor<Ty>> &U_0);
    }; // namespace GRQI
}; // namespace Algorithm

//#include "algorithm/tucker/hooi_als.tpp"

#endif