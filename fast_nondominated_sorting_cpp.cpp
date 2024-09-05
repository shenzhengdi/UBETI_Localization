#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <iostream>

namespace py = pybind11;


py::tuple generate_frontier_info(
    py::array_t<int>& frontier_sort_idx,
    py::array_t<int>& frontier_idx,
    py::array_t<int>& S,
    const py::array_t<bool>& is_first_dominating
) {
    auto frontier_sort_idx_buffer = frontier_sort_idx.request();
    int *frontier_sort_idx_ptr = (int *) frontier_sort_idx_buffer.ptr;
    size_t pop_size = frontier_sort_idx_buffer.shape[0];

    auto frontier_idx_buffer = frontier_idx.request();
    int *frontier_idx_ptr = (int *) frontier_idx_buffer.ptr;

    auto S_buffer = S.request();
    int *S_ptr = (int *) S_buffer.ptr;

    auto is_first_dominating_buffer = is_first_dominating.request();
    bool *is_first_dominating_ptr = (bool *) is_first_dominating_buffer.ptr;

    std::vector<size_t> Sp;

    for (size_t pop_idx = 0; pop_idx < pop_size; ++pop_idx) {
        Sp.clear();
        Sp.reserve(pop_size);

        for (size_t q = 0; q < pop_size; ++q){
            if (is_first_dominating_ptr[pop_idx * pop_size + q]) {
                Sp.push_back(q);
            }
            else if (is_first_dominating_ptr[q * pop_size + pop_idx]) {
                frontier_idx_ptr[pop_idx]++;
            }
            size_t S_start_idx = pop_idx * pop_size;
            for (size_t j = 0; j < Sp.size(); ++j) {
                S_ptr[S_start_idx + j] = Sp[j];
            }
        }

        if (frontier_idx_ptr[pop_idx] == 0) {
            frontier_sort_idx_ptr[pop_idx] = 1;

        }
    }

    py::tuple result(3);
    result[0] = S;
    result[1] = frontier_idx;
    result[2] = frontier_sort_idx;
    return result;
}


py::array_t<int> update_frontier_info(
    py::array_t<int>& frontier_sort_idx,
    py::array_t<int>& frontier_idx,
    py::array_t<int>& S
) {
    auto frontier_sort_idx_buffer = frontier_sort_idx.request();
    int *frontier_sort_idx_ptr = (int *) frontier_sort_idx_buffer.ptr;
    size_t pop_size = frontier_sort_idx_buffer.shape[0];

    auto frontier_idx_buffer = frontier_idx.request();
    int *frontier_idx_ptr = (int *) frontier_idx_buffer.ptr;

    auto S_buffer = S.request();
    int *S_ptr = (int *) S_buffer.ptr;

    int frontier_i = 1;
    bool should_continue = true;
    while (should_continue) {
        should_continue = false;
        for (size_t p = 0; p < pop_size; ++p) {
            if (frontier_sort_idx_ptr[p] == frontier_i) {
                should_continue = true;
                for (size_t q_idx = 0; q_idx < pop_size; ++q_idx) {
                    auto q = S_ptr[p * pop_size + q_idx];
                    if (q != -1) {
                        frontier_idx_ptr[q]--;
                        if (frontier_idx_ptr[q] == 0) {
                            frontier_sort_idx_ptr[q] = frontier_i + 1;
                        }
                    }
                }
            }
        }
        frontier_i++;
    }
    return frontier_sort_idx;
}


PYBIND11_MODULE(fast_nondominated_sorting_cpp, m) {
    m.def("generate_frontier_info", &generate_frontier_info, "generate_frontier_info");
    m.def("update_frontier_info", &update_frontier_info, "update_frontier_info");
}
