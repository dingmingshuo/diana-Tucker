//
// Created by 丁明朔 on 2022/1/22.
//

#include "communicator.hpp"
#include "tensor.hpp"
#include "distribution.hpp"
#include "logger.hpp"
#include "algorithm.hpp"
#include "summary.hpp"
#include <fstream>

int main(int argc, char *argv[]) {
    mpi_init(argc, argv);
    srand((unsigned int) 20000905);
    std::ifstream fin(argv[1]);

    // Init shape
    size_t N;
    shape_t I, R, par;
    fin >> N;
    for (size_t i = 0; i < N; i++) {
        size_t i_now, r_now, par_now;
        fin >> i_now >> r_now >> par_now;
        I.push_back(i_now);
        R.push_back(r_now);
        par.push_back(par_now);
    }

    // Init distribution and tensor
    auto *distribution =
            new DistributionCartesianBlock(par, mpi_rank());
    auto T = Tensor<double>(distribution, I);
    T.randn();

    // Calculate
    Summary::init();
    auto[G, U] = Algorithm::Tucker::HOOI_ALS(T, R, 5);
    Summary::finalize();

    // Pring summary
    Summary::print_summary();
    MPI_Finalize();
    return 0;
}