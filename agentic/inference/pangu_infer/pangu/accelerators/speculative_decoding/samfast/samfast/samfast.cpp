// Build with: python -m pip install -e .
// Then: from samfast import SAM

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include <unordered_map>
#include <string>
#include <cstdint>
#include <stdexcept>
#include <utility>

namespace py = pybind11;

struct State {
    std::unordered_map<int32_t, int32_t> next; // token -> state index
    int32_t link = -1;
    int32_t length = 0;
    int32_t min_endpos = 0;
};

class SAMCpp {
public:
    explicit SAMCpp(int32_t n_predicts = 3, const std::string& device = "cpu")
        : n_predicts_(n_predicts), device_(device) {
        reset();
    }

    void reset() {
        states_.clear();
        states_.push_back(State{}); // state 0
        states_[0].link = -1;
        states_[0].length = 0;
        states_[0].min_endpos = 0;
        input_ids_.clear();
        input_ids_.push_back(-1);
        last_ = 0;
        max_length_ = 0;
        cur_index_ = 0;
        cur_length_ = 0;
    }

    // Add a single token to SAM (same semantics as Python add_state)
    void add_state(int32_t token) {
        max_length_ += 1;
        int32_t cur = expand_state(State{ /*next*/{}, /*link*/-1, /*length*/max_length_, /*min_endpos*/max_length_ });
        int32_t p = last_;
        while (p != -1 && states_[p].next.find(token) == states_[p].next.end()) {
            states_[p].next[token] = cur;
            p = states_[p].link;
        }
        if (p == -1) {
            states_[cur].link = 0;
        } else {
            int32_t q = states_[p].next[token];
            if (states_[p].length + 1 == states_[q].length) {
                states_[cur].link = q;
            } else {
                // clone q
                State clone = states_[q]; // deep copy
                int32_t clone_idx = expand_state(clone);
                states_[clone_idx].length = states_[p].length + 1;
                while (p != -1 && states_[p].next[token] == q) {
                    states_[p].next[token] = clone_idx;
                    p = states_[p].link;
                }
                states_[q].link = states_[cur].link = clone_idx;
            }
        }
        last_ = cur;
    }

    // Transfer function used by lookup/transfer_cur_state
    std::pair<int32_t,int32_t> transfer_state(int32_t index, int32_t length, int32_t token) const {
        while (index != 0 && states_[index].next.find(token) == states_[index].next.end()) {
            index = states_[index].link;
            length = states_[index].length;
        }
        auto it = states_[index].next.find(token);
        if (it != states_[index].next.end()) {
            index = it->second;
            length += 1;
        } else {
            index = 0;
            length = 0;
        }
        return {index, length};
    }

    void transfer_cur_state(int32_t token) {
        auto r = transfer_state(cur_index_, cur_length_, token);
        cur_index_ = r.first;
        cur_length_ = r.second;
    }

    // Add a sequence of tokens (will also advance cur state like Python version)
    void add_tokens(py::array_t<int32_t, py::array::c_style | py::array::forcecast> tokens) {
        py::gil_scoped_release release; // heavy loop: release GIL
        auto buf = tokens.request();
        auto* ptr = static_cast<int32_t*>(buf.ptr);
        const size_t n = static_cast<size_t>(buf.size);
        for (size_t i = 0; i < n; ++i) {
            transfer_cur_state(ptr[i]);
            add_state(ptr[i]);
        }
        // append to input_ids_
        input_ids_.reserve(input_ids_.size() + n);
        for (size_t i = 0; i < n; ++i) input_ids_.push_back(ptr[i]);
    }

    // Only transfer (no state creation)
    void transfer_tokens(py::array_t<int32_t, py::array::c_style | py::array::forcecast> tokens) {
        py::gil_scoped_release release; // release GIL
        auto buf = tokens.request();
        auto* ptr = static_cast<int32_t*>(buf.ptr);
        const size_t n = static_cast<size_t>(buf.size);
        for (size_t i = 0; i < n; ++i) {
            transfer_cur_state(ptr[i]);
        }
    }

    std::pair<int32_t,int32_t> lookup(int32_t token) const {
        return transfer_state(cur_index_, cur_length_, token);
    }

    int32_t to_anc(int32_t index) const {
        if (index != 0) {
            int32_t length_to_end = max_length_ - states_[index].min_endpos;
            while (states_[index].link != 0 && n_predicts_ > length_to_end) {
                index = states_[index].link;
                length_to_end = max_length_ - states_[index].min_endpos;
            }
        }
        return index;
    }

    // gen_draft per Python semantics
    std::vector<int32_t> gen_draft(int32_t index, int32_t start_token) const {
        (void)start_token; // unused in current logic (kept for API compatibility)
        index = to_anc(index);
        int32_t endpos = states_[index].min_endpos;
        std::vector<int32_t> pred;
        if (endpos != 0) {
            int32_t from = endpos + 1;
            int32_t to = std::min<int32_t>(static_cast<int32_t>(input_ids_.size()), endpos + n_predicts_ + 1);
            for (int32_t i = from; i < to; ++i) pred.push_back(input_ids_[i]);
        }
        return pred;
    }

    // propose(context): rebuild from context[:-1], then lookup on last, then gen_draft
    py::array_t<int32_t> propose(py::array_t<int32_t, py::array::c_style | py::array::forcecast> context) {
        py::gil_scoped_release release; // encompass heavy work
        auto buf = context.request();
        auto* ptr = static_cast<int32_t*>(buf.ptr);
        const int64_t n = static_cast<int64_t>(buf.size);
        if (n <= 0) {
            return py::array_t<int32_t>(0);
        }
        reset();
        // add context[:-1]
        for (int64_t i = 0; i < n - 1; ++i) {
            transfer_cur_state(ptr[i]);
            add_state(ptr[i]);
            input_ids_.push_back(ptr[i]);
        }
        int32_t query_token = ptr[n - 1];
        auto r = lookup(query_token);
        auto seq = gen_draft(r.first, query_token);
        // return as numpy array
        py::gil_scoped_acquire acquire; // to create Python objects safely
        return py::array_t<int32_t>(seq.size(), seq.data());
    }

    // Expose a few accessors for completeness
    int32_t n_predicts() const { return n_predicts_; }
    std::string device() const { return device_; }

private:
    int32_t expand_state(const State& st) {
        states_.push_back(st);
        return static_cast<int32_t>(states_.size() - 1);
    }

    // data
    int32_t n_predicts_ = 3;
    std::string device_;

    std::vector<State> states_;
    std::vector<int32_t> input_ids_;

    int32_t last_ = 0;
    int32_t max_length_ = 0;
    int32_t cur_index_ = 0;
    int32_t cur_length_ = 0;
};

PYBIND11_MODULE(_samfast, m) {
    m.doc() = "GIL-free Suffix Automaton (SAM) via pybind11, API compatible with the original Python class";

    py::class_<SAMCpp>(m, "SAM")
        .def(py::init<int32_t, const std::string&>(), py::arg("n_predicts")=3, py::arg("device")="cpu")
        .def("reset", &SAMCpp::reset, "Reset automaton state")
        .def("add_state", &SAMCpp::add_state, py::arg("token"), "Add a single token and update links")
        .def("transfer_state", &SAMCpp::transfer_state, py::arg("index"), py::arg("length"), py::arg("token"))
        .def("transfer_cur_state", &SAMCpp::transfer_cur_state, py::arg("token"))
        .def("add_tokens", &SAMCpp::add_tokens, py::arg("tokens"), "Advance and add states for each token (releases GIL)")
        .def("transfer_tokens", &SAMCpp::transfer_tokens, py::arg("tokens"), "Only advance current state (releases GIL)")
        .def("lookup", &SAMCpp::lookup, py::arg("token"), "Lookup next index/length from current state")
        .def("to_anc", &SAMCpp::to_anc, py::arg("index"), "Lift to ancestor based on n_predicts window")
        .def("gen_draft", &SAMCpp::gen_draft, py::arg("index"), py::arg("start_token"), "Generate draft sequence per Python version")
        .def("propose", &SAMCpp::propose, py::arg("context_token_ids"), "Compute draft for context (releases GIL)") 
        .def("n_predicts", &SAMCpp::n_predicts)
        .def("device", &SAMCpp::device);
}
