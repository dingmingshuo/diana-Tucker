//
// Created by 丁明朔 on 2022/1/22.
//

#include "communicator.hpp"
#include "tensor.hpp"
#include "distribution.hpp"
#include "logger.hpp"
#include "summary.hpp"

int main() {
    mpi_init();
    Summary::init();
    srand(mpi_rank());
    shape_t shape{500, 400, 300};
    shape_t par{2, 3, 1};
    DistributionCartesianBlock *distribution =
            new DistributionCartesianBlock(par, mpi_rank());
    DistributionGlobal *dis_g = new DistributionGlobal();
    Tensor<double> t(distribution, shape);
    for (size_t i = 0; i < t.size(); i++) {
        t[i] = 10.0 * mpi_rank() + 1.0 * i;
    }
    Tensor<double> m(dis_g, {330, 400});
    for (size_t i = 0; i < m.size(); i++) {
        m[i] = (double) i;
    }
    auto t_new = Function::ttm(t, m, 1);
    if (mpi_rank() == 0) {
        print_vec(t_new.shape_global());
    }
    Summary::finalize();
    Summary::print_summary();
    MPI_Finalize();
    return 0;
}