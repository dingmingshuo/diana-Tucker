#include "communicator.hpp"

void mpi_init() { MPI_Init(nullptr, nullptr); }

void mpi_init(int argc, char **argv) { MPI_Init(&argc, &argv); }

int mpi_rank() {
    int ret;
    MPI_Comm_rank(MPI_COMM_WORLD, &ret);
    return ret;
}

int mpi_size() {
    int ret;
    MPI_Comm_size(MPI_COMM_WORLD, &ret);
    return ret;
}