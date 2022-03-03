#include "distribution.hpp"
#include "def.hpp"
#include "logger.hpp"

void distribution_assert_valid_input_(shape_t global_shape, shape_t local_shape,
                                      shape_t local_start = shape_t(),
                                      shape_t local_end = shape_t()) {
#ifndef PERFORMANCE_MODE
    assert(global_shape.size() != 0);
    assert(local_shape.size() == 0);
    assert(local_start.size() == 0);
    assert(local_end.size() == 0);
    for (auto dim : global_shape) {
        assert(dim > 0);
    }
#endif
}

Distribution::Distribution() {}

Distribution::Distribution(Distribution::Type type) { this->type_ = type; }

Distribution::Type Distribution::type() const { return this->type_; }

void Distribution::get_local_data(shape_t global_shape, shape_t &local_shape,
                                  shape_t &local_start, shape_t &local_end) {
    distribution_assert_valid_input_(global_shape, local_shape, local_start,
                                     local_end);
    for (auto dim : global_shape) {
        local_shape.push_back(dim);
        local_start.push_back(0);
        local_end.push_back(dim);
    }
}

void Distribution::get_local_data(int rank, shape_t global_shape,
                                  shape_t &local_shape, shape_t &local_start,
                                  shape_t &local_end) {
    DIANA_UNUSED(rank);
    return this->get_local_data(global_shape, local_shape, local_start,
                                local_end);
}

void Distribution::get_local_shape(shape_t global_shape, shape_t &local_shape) {
    distribution_assert_valid_input_(global_shape, local_shape);
    for (auto dim : global_shape) {
        local_shape.push_back(dim);
    }
}

void Distribution::get_local_shape(int rank, shape_t global_shape,
                                   shape_t &local_shape) {
    DIANA_UNUSED(rank);
    return this->get_local_shape(global_shape, local_shape);
}

size_t Distribution::global_size(shape_t global_shape) {
    size_t size = 1;
    for (size_t dim : global_shape) {
        size *= dim;
    }
    return size;
}

size_t Distribution::global_size(int rank, shape_t global_shape) {
    DIANA_UNUSED(rank);
    return this->global_size(global_shape);
}

size_t Distribution::local_size(shape_t local_shape) {
    size_t size = 1;
    for (size_t dim : local_shape) {
        size *= dim;
    }
    return size;
}

size_t Distribution::local_size(int rank, shape_t local_shape) {
    DIANA_UNUSED(rank);
    return this->local_size(local_shape);
}

DistributionLocal::DistributionLocal()
    : Distribution(Distribution::Type::kLocal) {}

DistributionGlobal::DistributionGlobal()
    : Distribution(Distribution::Type::kGlobal) {}

DistributionCartesianBlock::DistributionCartesianBlock(shape_t partition,
                                                       int rank)
    : Distribution(Distribution::Type::kCartesianBlock) {
    this->ndim_ = partition.size();
    this->partition_.assign(partition.begin(), partition.end());
    this->coordinate_ = shape_t();
    for (auto item : partition) {
        this->coordinate_.push_back(rank % item);
        rank /= (int)item;
    }
    assert(this->ndim_ == this->coordinate_.size());
    for (size_t i = 0; i < this->ndim_; i++) {
        assert(this->coordinate_[i] < this->partition_[i]);
        assert(this->partition_[i] > 0);
    }
}

shape_t DistributionCartesianBlock::partition() const {
    return this->partition_;
}

shape_t DistributionCartesianBlock::coordinate() const {
    return this->coordinate_;
}

shape_t DistributionCartesianBlock::coordinate(int rank) const {
    shape_t coordinate;
    for (auto item : this->partition_) {
        coordinate.push_back(rank % item);
        rank /= (int)item;
    }
    return coordinate;
}

size_t DistributionCartesianBlock::ndim() const { return this->ndim_; }

void DistributionCartesianBlock::get_local_data(shape_t global_shape,
                                                shape_t &local_shape,
                                                shape_t &local_start,
                                                shape_t &local_end) {
    distribution_assert_valid_input_(global_shape, local_shape, local_start,
                                     local_end);
    for (size_t i = 0; i < this->ndim_; i++) {
        size_t start = DIANA_CEILDIV(global_shape[i] * this->coordinate_[i],
                                     this->partition_[i]);
        size_t end = DIANA_CEILDIV(global_shape[i] * (this->coordinate_[i] + 1),
                                   this->partition_[i]);
        local_start.push_back(start);
        local_end.push_back(end);
        local_shape.push_back(end - start);
    }
}

void DistributionCartesianBlock::get_local_data(int rank, shape_t global_shape,
                                                shape_t &local_shape,
                                                shape_t &local_start,
                                                shape_t &local_end) {
    shape_t coord = this->coordinate(rank);
    distribution_assert_valid_input_(global_shape, local_shape, local_start,
                                     local_end);
    for (size_t i = 0; i < this->ndim_; i++) {
        size_t start =
            DIANA_CEILDIV(global_shape[i] * coord[i], this->partition_[i]);
        size_t end = DIANA_CEILDIV(global_shape[i] * (coord[i] + 1),
                                   this->partition_[i]);
        local_start.push_back(start);
        local_end.push_back(end);
        local_shape.push_back(end - start);
    }
}

void DistributionCartesianBlock::get_local_shape(shape_t global_shape,
                                                 shape_t &local_shape) {
    distribution_assert_valid_input_(global_shape, local_shape);
    for (size_t i = 0; i < this->ndim_; i++) {
        size_t start = DIANA_CEILDIV(global_shape[i] * this->coordinate_[i],
                                     this->partition_[i]);
        size_t end = DIANA_CEILDIV(global_shape[i] * (this->coordinate_[i] + 1),
                                   this->partition_[i]);
        local_shape.push_back(end - start);
    }
}

void DistributionCartesianBlock::get_local_shape(int rank, shape_t global_shape,
                                                 shape_t &local_shape) {
    shape_t coord = this->coordinate(rank);
    distribution_assert_valid_input_(global_shape, local_shape);
    for (size_t i = 0; i < this->ndim_; i++) {
        size_t start =
            DIANA_CEILDIV(global_shape[i] * coord[i], this->partition_[i]);
        size_t end = DIANA_CEILDIV(global_shape[i] * (coord[i] + 1),
                                   this->partition_[i]);
        local_shape.push_back(end - start);
    }
}

size_t DistributionCartesianBlock::local_size(shape_t global_shape) {
    size_t size = 1;
    for (size_t i = 0; i < this->ndim_; i++) {
        size_t start = DIANA_CEILDIV(global_shape[i] * this->coordinate_[i],
                                     this->partition_[i]);
        size_t end = DIANA_CEILDIV(global_shape[i] * (this->coordinate_[i] + 1),
                                   this->partition_[i]);
        size *= (end - start);
    }
    return size;
}

size_t DistributionCartesianBlock::local_size(int rank, shape_t global_shape) {
    shape_t coord = this->coordinate(rank);
    size_t size = 1;
    for (size_t i = 0; i < this->ndim_; i++) {
        size_t start =
            DIANA_CEILDIV(global_shape[i] * coord[i], this->partition_[i]);
        size_t end = DIANA_CEILDIV(global_shape[i] * (coord[i] + 1),
                                   this->partition_[i]);
        size *= (end - start);
    }
    return size;
}