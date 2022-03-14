//
// Created by 丁明朔 on 2022/1/22.
//

#include "communicator.hpp"
#include "tensor.hpp"
#include "distribution.hpp"
#include "logger.hpp"
#include "algorithm.hpp"
#include "summary.hpp"

int main() {
    mpi_init();
    srand((unsigned int) 20000905);
    shape_t shape{500, 400, 300};
    shape_t par{3, 2, 1};
    auto *distribution =
            new DistributionCartesianBlock(par, mpi_rank());
    auto T = Tensor<double>(distribution, shape);
    T.randn();
    auto[G, U] = Algorithm::Tucker::HOOI_ALS(T, {50, 30, 20}, 10);
    MPI_Finalize();
    return 0;
}