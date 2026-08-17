// SPDX-FileCopyrightText: 2024 PairInteraction Developers
// SPDX-License-Identifier: LGPL-3.0-or-later

#pragma once

#include "pairinteraction/utils/eigen_assertion.hpp"

#include <Eigen/Core>
#include <initializer_list>
#include <set>
#include <vector>

namespace pairinteraction {
enum class SorterType : unsigned char;

struct IndicesOfBlock {
    IndicesOfBlock(size_t start, size_t end);
    size_t size() const;
    size_t start;
    size_t end;
};

class IndicesOfBlocksCreator {
public:
    IndicesOfBlocksCreator(std::initializer_list<size_t> boundaries);
    void add(size_t boundary);
    std::vector<IndicesOfBlock> create() const;
    size_t size() const;

private:
    std::set<size_t> boundaries;
};

class SorterBuilderInterface {
public:
    virtual ~SorterBuilderInterface() = default;
    virtual Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic>
    get_sorter(const std::vector<SorterType> &labels) const = 0;
    virtual std::vector<IndicesOfBlock>
    get_indices_of_blocks(const std::vector<SorterType> &labels) const = 0;
};
} // namespace pairinteraction
