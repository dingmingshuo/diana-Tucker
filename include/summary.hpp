//
// Created by 丁明朔 on 2022/3/3.
//

#ifndef DIANA_TUCKER_SUMMARY_HPP
#define DIANA_TUCKER_SUMMARY_HPP

#include <string>
#include <vector>
#include <mpi.h>
#include <cmath>
#include <map>

#include "def.hpp"

class Summary {
private:
    struct Event {
        std::string name;

        size_t caller_id;

        long long flop;
        long long bandwidth;

        double time_start;
        double time_end;
        double time_length;
        double time_counted;

        std::vector<size_t> callee_ids;
    };
    static std::vector<Event> events_;
    static std::map<std::string, std::vector<size_t>> events_name_map_;
    static size_t last_id_;

public:

    static void start(const std::string &name, long long flop = 0,
                      long long bandwidth = 0);

    static void end(const std::string &name);


    static void print_summary();
};

#endif //DIANA_TUCKER_SUMMARY_HPP
