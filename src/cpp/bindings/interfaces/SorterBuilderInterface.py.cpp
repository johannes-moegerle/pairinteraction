// SPDX-FileCopyrightText: 2024 PairInteraction Developers
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "./SorterBuilderInterface.py.hpp"

#include "pairinteraction/interfaces/SorterBuilderInterface.hpp"

#include <nanobind/eigen/dense.h>
#include <nanobind/eigen/sparse.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/complex.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;
using namespace nb::literals;
using namespace pairinteraction;

static void declare_permutation_matrix(nb::module_ &m) {
    nb::class_<Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic>> pyclass(
        m, "PermutationMatrix");
    pyclass.def(nb::init<>());
}

static void declare_indices_of_blocks(nb::module_ &m) {
    nb::class_<IndicesOfBlock> pyclass(m, "IndicesOfBlock");
    pyclass.def(nb::init<size_t, size_t>(), "start"_a, "end"_a)
        .def_rw("start", &IndicesOfBlock::start)
        .def_rw("end", &IndicesOfBlock::end);
}

static void declare_indices_of_blocks_creator(nb::module_ &m) {
    nb::class_<IndicesOfBlocksCreator> pyclass(m, "IndicesOfBlocksCreator");
}

template <typename T>
static void declare_sorter_builder_interface(nb::module_ &m, std::string const &type_name) {
    std::string pyclass_name = "SorterBuilderInterface" + type_name;
    nb::class_<SorterBuilderInterface<T>> pyclass(m, pyclass_name.c_str());
}

void bind_sorter_builder_interface(nb::module_ &m) {
    declare_permutation_matrix(m);
    declare_indices_of_blocks(m);
    declare_indices_of_blocks_creator(m);
    declare_sorter_builder_interface<double>(m, "Real");
    declare_sorter_builder_interface<std::complex<double>>(m, "Complex");
}
