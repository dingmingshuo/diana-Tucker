#include "def.hpp"
#include "logger.hpp"
#include "summary.hpp"

template<class Ty>
Communicator<Ty>::Communicator() {
    MPI_Comm_size(MPI_COMM_WORLD, &this->size_);
    MPI_Comm_rank(MPI_COMM_WORLD, &this->rank_);
    if constexpr (std::is_same<Ty, float32>::value) {
        this->mpi_type_ = MPI_FLOAT;
    } else if constexpr (std::is_same<Ty, float64>::value) {
        this->mpi_type_ = MPI_DOUBLE;
    } else if constexpr (std::is_same<Ty, complex32>::value) {
        this->mpi_type_ = MPI_C_COMPLEX;
    } else if constexpr (std::is_same<Ty, complex64>::value) {
        this->mpi_type_ = MPI_C_DOUBLE_COMPLEX;
    }
}

template<class Ty>
Communicator<Ty>::~Communicator() {}

template<class Ty>
int Communicator<Ty>::size() const { return this->size_; }

template<class Ty>
int Communicator<Ty>::rank() const { return this->rank_; }

template<class Ty>
MPI_Datatype Communicator<Ty>::mpi_type() const {
    return this->mpi_type_;
}

template<class Ty>
MPI_Comm Communicator<Ty>::comm_split(int color, int rank, MPI_Comm comm) {
    Summary::start(__func__);
    MPI_Comm ret;
    MPI_Comm_split(comm, color, rank, &ret);
    Summary::end(__func__);
    return ret;
}

template<class Ty>
void Communicator<Ty>::bcast(Ty *A, int size, int proc, MPI_Comm comm) {
    Summary::start(__func__);
    MPI_Bcast(A, size, this->mpi_type_, proc, comm);
    Summary::end(__func__);
}

template<class Ty>
void Communicator<Ty>::allreduce_inplace(Ty *A, int size, MPI_Op op,
                                         MPI_Comm comm) {
    Summary::start(__func__);
    MPI_Allreduce(MPI_IN_PLACE, A, size, this->mpi_type_, op, comm);
    Summary::end(__func__);
}

template<class Ty>
void Communicator<Ty>::sendrecv(Ty *A, int size, int des, MPI_Comm comm) {
    Summary::start(__func__);
    MPI_Status status;
    MPI_Sendrecv_replace(A, size, this->mpi_type_, des, 0, des, 0, comm,
                         &status);
    Summary::end(__func__);
}

template<class Ty>
void Communicator<Ty>::reduce_scatter(Ty *sendbuf, Ty *recvbuf,
                                      const int *recvcounts, MPI_Op op,
                                      MPI_Comm comm) {
    Summary::start(__func__);
    MPI_Reduce_scatter(sendbuf, recvbuf, recvcounts, this->mpi_type_, op, comm);
    Summary::end(__func__);
}

template<class Ty>
void Communicator<Ty>::gatherv(Ty *sendbuf, int sendcount, Ty *recvbuf,
                               const int *recvcounts, const int *displs,
                               int root, MPI_Comm comm) {
    Summary::start(__func__);
    MPI_Gatherv(sendbuf, sendcount, this->mpi_type_, recvbuf, recvcounts,
                displs, this->mpi_type_, root, comm);
    Summary::end(__func__);
}

template<class Ty>
void Communicator<Ty>::scatterv(Ty *sendbuf, const int *sendcounts,
                                const int *displs, Ty *recvbuf, int recvcount,
                                int root, MPI_Comm comm) {
    Summary::start(__func__);
    MPI_Scatterv(sendbuf, sendcounts, displs, this->mpi_type_, recvbuf,
                 recvcount, this->mpi_type_, root, comm);
    Summary::end(__func__);
}

template<class Ty>
void Communicator<Ty>::barrier(MPI_Comm comm) {
    Summary::start(__func__);
    MPI_Barrier(comm);
    Summary::end(__func__);
}