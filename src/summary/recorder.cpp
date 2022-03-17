//
// Created by 丁明朔 on 2022/3/17.
//

#include "summary.hpp"

Summary::Recorder::Recorder(const std::string &name) {
    this->name_ = name;
    Summary::start(name);
}

Summary::Recorder::Recorder(const std::string &name, long long flop) {
    this->name_ = name;
    Summary::start(name, flop);
}

Summary::Recorder::Recorder(const std::string &name, long long flop,
                            long long bandwidth) {
    this->name_ = name;
    Summary::start(name, flop, bandwidth);
}

Summary::Recorder::~Recorder() {
    Summary::end(this->name_);
}