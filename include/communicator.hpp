#ifndef __DIANA_CORE_INCLUDE_COMMUNICATOR_HPP__
#define __DIANA_CORE_INCLUDE_COMMUNICATOR_HPP__

#include <cstdlib>
#include <mpi.h>

void mpi_init();
int mpi_rank();
int mpi_size();

template <typename Ty> class Communicator {
  private:
    int rank_;
    int size_;
    MPI_Datatype mpi_type_;

  public:
    Communicator();
    ~Communicator();

    int rank() const;
    int size() const;
    MPI_Datatype mpi_type() const;
    MPI_Comm comm_split(int color, int rank, MPI_Comm comm = MPI_COMM_WORLD);

    void bcast(Ty *A, int size, int proc, MPI_Comm comm = MPI_COMM_WORLD);
    void allreduce_inplace(Ty *A, int size, MPI_Op op,
                           MPI_Comm comm = MPI_COMM_WORLD);
    void sendrecv(Ty *A, int size, int dest, MPI_Comm comm = MPI_COMM_WORLD);
    void reduce_scatter(Ty *sendbuf, Ty *recvbuf, const int *recvcounts,
                        MPI_Op op, MPI_Comm comm = MPI_COMM_WORLD);
    void gatherv(Ty *sendbuf, int sendcount, Ty *recvbuf, const int *recvcounts,
                 const int *displs, int root, MPI_Comm comm = MPI_COMM_WORLD);
    void scatterv(Ty *sendbuf, const int *sendcounts, const int *displs,
                  Ty *recvbuf, int recvcount, int root,
                  MPI_Comm comm = MPI_COMM_WORLD);
    void barrier(MPI_Comm comm = MPI_COMM_WORLD);
};

#include "communicator.tpp"

#endif