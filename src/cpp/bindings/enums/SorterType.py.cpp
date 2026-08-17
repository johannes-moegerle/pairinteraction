// SPDX-FileCopyrightText: 2024 PairInteraction Developers
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "./SorterType.py.hpp"

#include "pairinteraction/enums/SorterType.hpp"

#include <nanobind/nanobind.h>

namespace nb = nanobind;
using namespace pairinteraction;

void bind_sorter_type(nb::module_ &m) {
    nb::enum_<SorterType>(m, "SorterType")
        .value("SORT_BY_QUANTUM_NUMBER_F", SorterType::SORT_BY_QUANTUM_NUMBER_F)
        .value("SORT_BY_QUANTUM_NUMBER_M", SorterType::SORT_BY_QUANTUM_NUMBER_M)
        .value("SORT_BY_PARITY", SorterType::SORT_BY_PARITY)
        .value("SORT_BY_ENERGY", SorterType::SORT_BY_ENERGY);
}
