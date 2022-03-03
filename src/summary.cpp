//
// Created by 丁明朔 on 2022/3/3.
//

#include <queue>
#include <iostream>

#include "summary.hpp"
#include "logger.hpp"

#define ROOT_ID SIZE_MAX
std::vector<Summary::Event> Summary::events_ = std::vector<Summary::Event>();
std::map<std::string, std::vector<size_t>>  Summary::events_name_map_ =
        std::map<std::string, std::vector<size_t>>();
size_t Summary::last_id_ = ROOT_ID;

void
Summary::start(const std::string &name, long long flop, long long bandwidth) {
    Summary::events_.push_back({
                                       name,
                                       Summary::last_id_,
                                       flop,
                                       bandwidth,
                                       MPI_Wtime(),
                                       0,
                                       0,
                                       0,
                                       std::vector<size_t>()
                               });
    Summary::last_id_ = events_.size() - 1;
    Summary::events_name_map_[name].push_back(Summary::last_id_);
}

void
Summary::end(const std::string &name) {
    size_t idx = Summary::last_id_;
    Summary::Event &event = Summary::events_[idx];
    assert(event.name == name);
    event.time_end = MPI_Wtime();
    event.time_length = event.time_end - event.time_start;
    event.time_length += event.time_length;
    Summary::last_id_ = event.caller_id;
    if (Summary::last_id_ != ROOT_ID) {
        // If last_id_ is not root, then gather the data to the caller.
        Summary::Event &caller_event = Summary::events_[Summary::last_id_];
        caller_event.callee_ids.push_back(idx);
        caller_event.flop += event.flop;
        caller_event.bandwidth += event.bandwidth;
        caller_event.time_counted += event.time_length;
    }
}

void fill_space_(std::string &s, size_t len) {
    while (s.length() < len) {
        s += " ";
    }
}

void add_data_(std::string &output, const std::string &data, size_t len) {
    std::string tmp = data;
    fill_space_(tmp, len);
    output += tmp;
}

void Summary::print_summary() {
    const int kNameMaxLength = 15;
    const int kCaptionLength = 15;
    const int kFirstSectionLength = 25;
    const std::string kSecondSectionCaption[] = {"Time(s)", "Time C.(s)",
                                                 "Time C.(%)", "Number",
                                                 "Avg. Time(s)"};
    const std::string kThirdSectionCaption[] = {"GFlop/s", "Bandw.(GB/s)"};
    std::string output;
    // Display caption row.
    fill_space_(output, kFirstSectionLength);
    output += "| ";
    for (const std::string &caption: kSecondSectionCaption) {
        add_data_(output, caption, kCaptionLength);
    }
    output += "| ";
    for (const std::string &caption: kThirdSectionCaption) {
        add_data_(output, caption, kCaptionLength);
    }
    output += "\n";
    // Display events.
    for (const auto &event_list: Summary::events_name_map_) {
        // First section, contains name only.
        std::string name;
        name += event_list.first.substr(0, kNameMaxLength);
        if (event_list.first.length() > kNameMaxLength) {
            name += "...";
        }
        add_data_(output, name, kFirstSectionLength);
        output += "| ";
        // Get important statistics.
        double time_length_total = 0;
        double time_length_counted = 0;
        long long flop = 0;
        long long bandwidth = 0;
        size_t number = event_list.second.size();
        for (const auto &idx: event_list.second) {
            const auto &event = Summary::events_[idx];
            time_length_total += event.time_length;
            time_length_counted += event.time_counted;
            flop += event.flop;
            bandwidth += event.bandwidth;
        }
        // Second section, contains "Time", "Time Counted", "Time Counted%",
        // "Number", "Average Time".
        add_data_(output, std::to_string(time_length_total),
                  kCaptionLength);
        if (time_length_counted == 0) {
            add_data_(output, "Kernel", kCaptionLength);
            add_data_(output, "Kernel", kCaptionLength);

        } else {
            add_data_(output, std::to_string(time_length_counted),
                      kCaptionLength);
            add_data_(output,
                      std::to_string(
                              time_length_counted / time_length_total * 100),
                      kCaptionLength);
        }
        add_data_(output, std::to_string(number), kCaptionLength);
        add_data_(output, std::to_string(time_length_total / (double) number),
                  kCaptionLength);
        output += "| ";
        // Third section, contains flop/s, bandwidth/s.
        add_data_(output,
                  std::to_string((double) flop / 1e9 / time_length_total),
                  kCaptionLength);
        add_data_(output,
                  std::to_string(
                          (double) bandwidth / 1073741824 / time_length_total),
                  kCaptionLength);
        output += "\n";
    }
    std::cerr << output;
}

