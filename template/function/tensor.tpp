#include "tensor.hpp"
#include "logger.hpp"
#include "summary.hpp"

#include <cmath>

namespace Function {
    template<typename Ty>
    Tensor<Ty> inverse(const Tensor<Ty> &A) {
        if (A.distribution() == nullptr) {
            assert(A.is_matrix());
            assert(A.shape()[0] == A.shape()[1]);
            Tensor<Ty> ret(A.shape());
            A.op()->inverse(ret.data(), A.data(), A.shape()[0]);
            return ret;
        }
        error("Invalid input or not implemented yet.");
    }

    template<typename Ty>
    std::tuple<Tensor<Ty>, Tensor<Ty>> reduced_LQ(const Tensor<Ty> &A) {
        if (A.distribution() == nullptr) {
            assert(A.is_matrix());
            assert(A.shape()[0] <= A.shape()[1]);
            size_t m = A.shape()[0];
            size_t n = A.shape()[1];
            Tensor<Ty> L({m, m}, true);
            Tensor<Ty> Q({m, n}, true);
            A.op()->LQ(L.data(), Q.data(), A.data(), A.shape()[0],
                       A.shape()[1]);
            return std::make_tuple(L, Q);
        }
        error("Invalid input or not implemented yet.");
    }

    /**
     * @brief Calculate \f$ \bm{\mathcal{A}}_{(n)} \bm{\mathcal{A}}_{(n)}^T \f$,
     * where \f$ \bm{\mathcal{A}} \f$ is a tensor.
     * @tparam Ty
     * @param A
     * @param n
     * @return
     */
    template<typename Ty>
    Tensor<Ty> gram(const Tensor<Ty> &A, size_t n) {
        if (A.distribution()->type() == Distribution::Type::kCartesianBlock) {
            // Initialization.
            auto *distrib = (DistributionCartesianBlock *) A.distribution();
            shape_t par = distrib->partition();
            shape_t coord = distrib->coordinate();
            const size_t kParN = par[n];
            const size_t kGlobalShapeN = A.shape_global()[n];
            const size_t kLocalShapeN = A.shape()[n];
            // Split communicator.
            auto[new_color, new_rank] = distrib->process_fiber(n);
            MPI_Comm comm_fiber = A.comm()->comm_split(new_color, new_rank);
            // Allocate double buffer.
            size_t max_size = A.size();
            Ty *data_A = A.data();
            Ty *databuf[2]; // Double buffer
            Communicator<size_t>::allreduce_inplace(&max_size, 1,
                                                    MPI_MAX, comm_fiber);
            databuf[0] = A.op()->alloc(max_size);
            databuf[1] = A.op()->alloc(max_size);
            // Allocate gram_buffer.
            const size_t row_length = kLocalShapeN;
            const size_t col_length = A.size() / kLocalShapeN;
            size_t *all_row_length = Operator<size_t>::alloc(kParN);
            size_t *gram_buf_start = Operator<size_t>::alloc(kParN);
            size_t gram_buf_size;
            auto gram_buf_point = (size_t) new_rank;
            Communicator<size_t>::allgather(&row_length, 1, all_row_length,
                                            comm_fiber);
            gram_buf_start[0] = 0;
            for (size_t i = 1; i < kParN; i++) {
                gram_buf_start[i] =
                        gram_buf_start[i - 1] + all_row_length[i - 1];
            }
            gram_buf_size = row_length * kGlobalShapeN;
            Ty *gram_buf = A.op()->alloc(gram_buf_size);
            // Allocate A_buf.
            Ty *A_buf = A.op()->alloc(A.size());
            // Initialize data transpose.
            int send_to_proc_id =
                    ((int) new_rank - 1 + (int) kParN) % (int) kParN;
            int recv_from_proc_id = ((int) new_rank + 1) % (int) kParN;
            // Matricization
            A.op()->tenmat(databuf[0], data_A, A.shape(), n);
            A.op()->mcpy(A_buf, databuf[0], A.size());
            // Do gram
            MPI_Request *request_send = A.comm()->new_request();
            MPI_Request *request_recv = A.comm()->new_request();
            for (size_t i = 0; i < kParN; i++) {
                if (i != 0) {
                    A.comm()->wait(request_send);
                    A.comm()->wait(request_recv);
                }
                if (i != kParN - 1) {
                    A.comm()->isend(request_send, databuf[i % 2],
                                    (int) max_size,
                                    send_to_proc_id, comm_fiber);
                    A.comm()->irecv(request_recv,
                                    databuf[(i + 1) % 2],
                                    (int) max_size,
                                    recv_from_proc_id, comm_fiber);
                    send_to_proc_id =
                            ((int) send_to_proc_id - 1 + (int) kParN) %
                            (int) kParN;
                    recv_from_proc_id =
                            ((int) recv_from_proc_id + 1) %
                            (int) kParN;
                }
                A.op()->matmulNT(gram_buf +
                                 gram_buf_start[gram_buf_point] * kLocalShapeN,
                                 A_buf, databuf[i % 2], row_length,
                                 all_row_length[gram_buf_point], col_length);
                gram_buf_point = (gram_buf_point + 1) % kParN;
            }
            // Allreduce.
            MPI_Comm comm_line = A.comm()->comm_split(new_rank, new_color);
            A.comm()->allreduce_inplace(gram_buf, (int) gram_buf_size, MPI_SUM,
                                        comm_line);
            // Gather.
            Tensor<Ty> gram({kGlobalShapeN, kGlobalShapeN}, false);
            Ty *gram_data = gram.data();
            int *recvcount = Operator<int>::alloc(kParN);
            int *displs = Operator<int>::alloc(kParN);
            for (size_t i = 0; i < kParN; i++) {
                recvcount[i] = (int) all_row_length[i];
                displs[i] = (int) gram_buf_start[i];
            }
            for (size_t i = 0; i < kGlobalShapeN; i++) {
                A.comm()->allgatherv(gram_buf + i * kLocalShapeN,
                                     (int) kLocalShapeN,
                                     gram_data + i * kGlobalShapeN,
                                     recvcount, displs, comm_fiber);
            }
            // Free buffers.
            A.op()->free(databuf[0]);
            A.op()->free(databuf[1]);
            Operator<size_t>::free(all_row_length);
            Operator<size_t>::free(gram_buf_start);
            A.op()->free(gram_buf);
            A.op()->free(A_buf);
            A.comm()->free_request(request_send);
            A.comm()->free_request(request_recv);
            return gram;
        }
        error("Invalid input or not implemented yet.");
    }


    template<typename Ty>
    Tensor<Ty> gram(const Tensor<Ty> &A) {
        if (A.distribution() == nullptr) {
            assert(A.is_matrix());
            size_t M = A.shape()[0];
            size_t N = A.shape()[1];
            Tensor<Ty> ret({M, M}, false);
            A.op()->matmulNT(ret.data(), A, A, M, M, N);
            return ret;
        }
    }

    /**
     * @brief  Calculate \f$ \bm{\mathcal{A}} \times_n \bm{M} \f$, where
     * \f$ \bm{\mathcal{A}} \f$ is a tensor and \f$ \bm{M} \f$ is a matrix.
     *
     * @tparam Ty
     * @param A A matrix of shape \f$ I_1 \times \cdots \times I_N \f$
     * @param M A matrix of shape \f$ J_n \times I_n \f$
     * @param n Index for TTM routine.
     * @return Tensor<Ty>
     */
    template<typename Ty>
    Tensor<Ty> ttm(const Tensor<Ty> &A, const Tensor<Ty> &M, size_t n) {
        if (A.distribution()->type() ==
            Distribution::Type::kCartesianBlock &&
            M.distribution()->type() == Distribution::Type::kGlobal) {
            // Initialization
            Summary::start(METHOD_NAME);
            assert(M.is_matrix());
            assert(A.shape_global()[n] == M.shape()[1]);
            auto distrib = (DistributionCartesianBlock *) A.distribution();
            shape_t coord = distrib->coordinate();
            shape_t par = distrib->partition();
            size_t row_length = M.shape()[0];
            size_t col_length = M.shape()[1];
            size_t col_local = A.shape()[n];
            size_t col_begin = DIANA_CEILDIV(col_length * coord[n], par[n]);
            size_t col_end = DIANA_CEILDIV(col_length * (coord[n] + 1),
                                           par[n]);
            assert(col_end - col_begin == col_local);
            size_t remain_size = A.size() / col_local;
            Ty *data_A = A.data();
            Ty *data_M = M.data();
            Ty *data_B = A.op()->alloc(A.size());
            Ty *data_Anew = A.op()->alloc(row_length * remain_size);
            // Matricization
            A.op()->tenmatt(data_B, data_A, A.shape(), n);
            // Do TTM
            A.op()->matmulNT(data_Anew, data_B,
                             data_M + col_begin * row_length,
                             remain_size, row_length, col_local);
            // Split communicator
            auto[new_color, new_rank] = distrib->process_fiber(n);
            MPI_Comm comm_new = A.comm()->comm_split(new_color, new_rank);
            // Do reduce-scatter
            shape_t new_shape = A.shape_global();
            new_shape[n] = row_length;
            Tensor<Ty> ret(A.distribution(), new_shape, false);
            Ty *data_ret = ret.data();
            Ty *data_ret_buf = ret.op()->alloc(ret.size());
            int *recvcounts = new int[par[n]];
            for (size_t i = 0; i < par[n]; i++) {
                recvcounts[i] =
                        (int) DIANA_CEILDIV(row_length * (coord[i] + 1),
                                            par[n]) -
                        (int) DIANA_CEILDIV(row_length * coord[i], par[n]);
                recvcounts[i] *= (int) remain_size;
            }
            ret.comm()->reduce_scatter(data_Anew, data_ret_buf, recvcounts,
                                       MPI_SUM,
                                       comm_new);
            // Tensorization
            ret.op()->mattten(data_ret, data_ret_buf, ret.shape(), n);
            // Free spaces
            A.op()->free(data_B);
            A.op()->free(data_Anew);
            A.op()->free(data_ret_buf);
            Summary::end(METHOD_NAME);
            return ret;
        }
        error("Invalid input or not implemented yet.");
    }

    template<typename Ty>
    Tensor<Ty>
    ttmc(const Tensor<Ty> &A, const std::vector<Tensor<Ty>> &M,
         const std::vector<size_t> &idx) {
        assert(M.size() == idx.size());
        for (size_t i = 0; i < M.size(); i++) {
            A = ttm(A, M[i], idx[i]);
        }
        return A;
    }

    template<typename Ty>
    Tensor<Ty> gather(const Tensor<Ty> &A) {
        if (A.distribution() == nullptr) {
            error("This tensor is not distributed, cannot be gathered.");
        }
        if (A.distribution()->type() == Distribution::Type::kLocal) {
            error("Distribution of this tensor is Distribution::Type::kLocal, "
                  "cannot be gathered.");
        }
        if (A.distribution()->type() == Distribution::Type::kGlobal) {
            error("Distribution of this tensor is Distribution::Type::kGlobal, "
                  "there is no need to be gathered.");
        }
        if (A.distribution()->type() ==
            Distribution::Type::kCartesianBlock) {
            Summary::start(METHOD_NAME);
            const int kZERO = 0;
            Tensor<Ty> ret(A.shape_global(), false);
            if (A.comm()->rank() == kZERO) {
                const int kMPISize = mpi_size();
                int *recvcounts = new int[(size_t) kMPISize];
                int *displs = new int[(size_t) kMPISize];
                // Calculate recvcounts
                for (int i = 0; i < kMPISize; i++) {
                    recvcounts[i] =
                            (int) A.distribution()->local_size(i,
                                                               A.shape_global());
                }
                // Calculate displs
                displs[0] = 0;
                for (int i = 1; i < kMPISize; i++) {
                    displs[i] = displs[i - 1] + recvcounts[i - 1];
                }
                // Receive data
                A.comm()->gatherv(A.data(), (int) A.size(), ret.data(),
                                  recvcounts,
                                  displs, kZERO);
                // Reorder data
                ret.op()->reorder_from_gather_cartesian_block(
                        ret.data(), ret.shape(),
                        ((DistributionCartesianBlock *) A.distribution())->partition(),
                        displs);
                // Bcast data
                A.comm()->bcast(ret.data(), (int) ret.size(), kZERO);
            } else {
                // Send data
                A.comm()->gatherv(A.data(), (int) A.size(), nullptr,
                                  nullptr,
                                  nullptr, kZERO);
                // Receive data from Bcast
                A.comm()->bcast(ret.data(), (int) ret.size(), kZERO);
            }
            Summary::end(METHOD_NAME);
            return ret;
        }
        error("Invalid input or not implemented yet.");
    }

    template<typename Ty>
    Tensor<Ty>
    scatter(const Tensor<Ty> &A, Distribution *distribution, int proc) {
        if (A.distribution() != nullptr &&
            A.distribution()->type() ==
            Distribution::Type::kCartesianBlock) {
            error("Distribution of this tensor is "
                  "Distribution::Type::kCartesianBlock,  there is no need to be "
                  "scatterd.");
        }
        if (A.distribution() != nullptr &&
            A.distribution()->type() == Distribution::Type::kGlobal) {
            error("Distribution of this tensor is Distribution::Type::kGlobal, "
                  "there is no need to be scatterd.");
        }
        if ((A.distribution() == nullptr ||
             A.distribution()->type() == Distribution::Type::kLocal) &&
            distribution->type() == Distribution::Type::kCartesianBlock) {
            Summary::start(METHOD_NAME);
            Tensor<Ty> ret(distribution, A.shape(), false);
            if (mpi_rank() == proc) {
                const int kMPISize = mpi_size();
                int *sendcounts = new int[(size_t) kMPISize];
                int *displs = new int[(size_t) kMPISize];
                // Calculate sendcounts
                for (int i = 0; i < kMPISize; i++) {
                    sendcounts[i] = (int) distribution->local_size(i,
                                                                   A.shape());
                }
                // Calculate displs
                displs[0] = 0;
                for (int i = 1; i < kMPISize; i++) {
                    displs[i] = displs[i - 1] + sendcounts[i - 1];
                }
                // Reorder data
                A.op()->reorder_for_scatter_cartesian_block(
                        A.data(), A.shape(),
                        ((DistributionCartesianBlock *) distribution)->partition(),
                        displs);
                // Scatter data
                ret.comm()->scatterv(A.data(), sendcounts, displs,
                                     ret.data(),
                                     (int) ret.size(), proc);
                // Reorder back data
                tick;
                A.op()->reorder_from_gather_cartesian_block(
                        A.data(), A.shape(),
                        ((DistributionCartesianBlock *) distribution)->partition(),
                        displs);
            } else {
                // Receive data
                ret.comm()->scatterv(nullptr, nullptr, nullptr, ret.data(),
                                     (int) ret.size(), proc);
            }
            Summary::end(METHOD_NAME);
            return ret;
        }
        error("Invalid input or not implemented yet.");
    }

    template<typename Ty>
    double fnorm(const Tensor<Ty> &A) {
        if (A.distribution()->type() ==
            Distribution::Type::kCartesianBlock) {
            Summary::start(METHOD_NAME);
            double ret = A.op()->fnorm(A.data(), A.size());
            ret = ret * ret;
            A.comm()->allreduce_inplace(&ret, 1, MPI_SUM);
            Summary::end(METHOD_NAME);
            return sqrt(ret);
        } else {
            return A.op()->fnorm(A.data(), A.size());
        }
    }

    template<typename Ty>
    Ty sum(const Tensor<Ty> &A) {
        if (A.distribution()->type() ==
            Distribution::Type::kCartesianBlock) {
            Summary::start(METHOD_NAME);
            Ty ret = A.op()->sum(A.data(), A.size());
            A.comm()->allreduce_inplace(&ret, 1, MPI_SUM);
            Summary::end(METHOD_NAME);
            return ret;
        } else {
            return A.op()->sum(A.data(), A.size());
        }
    }

} // namespace Function