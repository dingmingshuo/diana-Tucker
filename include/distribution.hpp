#ifndef __DIANA_CORE_DISTRIBUTION_TENSOR_HPP__
#define __DIANA_CORE_DISTRIBUTION_TENSOR_HPP__

#include "def.hpp"

/**
 * @enum Distribution
 * @brief Enumerate class of tensor's distribution.
 *
 * Distribution enumerate class using to indicate multiple types of tensor
 * distribution.
 */
class Distribution {
  public:
    enum Type : int {
        kLocal,  /**< this tensor is only stored on local process. */
        kGlobal, /**< a redundant copy of this tensor is stored on each process.
                  */
        kCartesianBlock, /**< this tensor is blockly stored on cartesian
                            processes. */
    };

  private:
    Type type_;

  public:
    Distribution();
    Distribution(Distribution::Type type);
    Distribution::Type type() const;

    virtual void get_local_data(shape_t global_shape, shape_t &local_shape,
                                shape_t &local_start, shape_t &local_end);
    virtual void get_local_data(int rank, shape_t global_shape,
                                shape_t &local_shape, shape_t &local_start,
                                shape_t &local_end);
    virtual void get_local_shape(shape_t global_shape, shape_t &local_shape);
    virtual void get_local_shape(int rank, shape_t global_shape,
                                 shape_t &local_shape);
    virtual size_t global_size(shape_t global_shape);
    virtual size_t global_size(int rank, shape_t global_shape);
    virtual size_t local_size(shape_t global_shape);
    virtual size_t local_size(int rank, shape_t global_shape);
};

class DistributionLocal : public Distribution {
  public:
    DistributionLocal();
};

class DistributionGlobal : public Distribution {
  public:
    DistributionGlobal();
};

class DistributionCartesianBlock : public Distribution {
  private:
    shape_t partition_;
    shape_t coordinate_;
    size_t ndim_;

  public:
    DistributionCartesianBlock(shape_t partition, int rank);
    shape_t partition() const;
    shape_t coordinate() const;
    shape_t coordinate(int rank) const;
    size_t ndim() const;

    virtual void get_local_data(shape_t global_shape, shape_t &local_shape,
                                shape_t &local_start, shape_t &local_end);
    virtual void get_local_data(int rank, shape_t global_shape,
                                shape_t &local_shape, shape_t &local_start,
                                shape_t &local_end);
    virtual void get_local_shape(shape_t global_shape, shape_t &local_shape);
    virtual void get_local_shape(int rank, shape_t global_shape,
                                 shape_t &local_shape);
    virtual size_t local_size(shape_t global_shape);
    virtual size_t local_size(int rank, shape_t global_shape);
};

#endif