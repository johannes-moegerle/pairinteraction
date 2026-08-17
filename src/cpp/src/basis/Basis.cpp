// SPDX-FileCopyrightText: 2024 PairInteraction Developers
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "pairinteraction/basis/Basis.hpp"

#include "pairinteraction/basis/BasisAtom.hpp"
#include "pairinteraction/basis/BasisPair.hpp"
#include "pairinteraction/enums/Parity.hpp"
#include "pairinteraction/enums/TransformationType.hpp"
#include "pairinteraction/ket/KetAtom.hpp"
#include "pairinteraction/ket/KetPair.hpp"
#include "pairinteraction/utils/TaskControl.hpp"
#include "pairinteraction/utils/eigen_assertion.hpp"
#include "pairinteraction/utils/eigen_compat.hpp"

#include <algorithm>
#include <cassert>
#include <set>
#include <tuple>
#include <vector>

namespace pairinteraction {

template <typename Scalar>
class BasisAtom;

template <typename Derived>
void Basis<Derived>::perform_blocks_checks(
    const std::set<TransformationType> &unique_labels) const {
    constexpr real_t numerical_precision = 100 * std::numeric_limits<real_t>::epsilon();

    // Blocks of states of equal energy would only be blocks of an already diagonal Hamiltonian,
    // which does not need to be diagonalized block by block anymore.
    if (unique_labels.contains(TransformationType::SORT_BY_ENERGY)) {
        throw std::invalid_argument("Blocks cannot be obtained by the energy. Note that sorting "
                                    "a system by the energy is supported nevertheless.");
    }

    // Check if the states are labeled by the requested labels
    const bool by_f = unique_labels.contains(TransformationType::SORT_BY_QUANTUM_NUMBER_F);
    const bool by_m = unique_labels.contains(TransformationType::SORT_BY_QUANTUM_NUMBER_M);
    const bool by_parity = unique_labels.contains(TransformationType::SORT_BY_PARITY);

    if (by_f && !_has_quantum_number_f) {
        throw std::invalid_argument(
            "States cannot be labeled and thus not sorted by the quantum number f.");
    }
    if (by_m && !_has_quantum_number_m) {
        throw std::invalid_argument(
            "States cannot be labeled and thus not sorted by the quantum number m.");
    }
    if (by_parity && !_has_parity) {
        throw std::invalid_argument("States cannot be labeled and thus not sorted by the parity.");
    }

    // Check if the states are sorted by the requested labels, i.e., if states that share the same
    // labels are contiguous. Otherwise, states belonging to the same block would be scattered over
    // several blocks and couplings between them would be lost.
    using label_t = std::tuple<real_t, real_t, Parity>;
    auto label_of = [&](Eigen::Index i) {
        return label_t{by_f ? state_index_to_quantum_number_f[i] : 0,
                       by_m ? state_index_to_quantum_number_m[i] : 0,
                       by_parity ? state_index_to_parity[i] : Parity::UNKNOWN};
    };
    auto is_equal = [](const label_t &a, const label_t &b) {
        return std::abs(std::get<0>(a) - std::get<0>(b)) <= numerical_precision &&
            std::abs(std::get<1>(a) - std::get<1>(b)) <= numerical_precision &&
            std::get<2>(a) == std::get<2>(b);
    };

    // Collect the labels of the contiguous blocks of states
    std::vector<label_t> labels_of_blocks;
    for (Eigen::Index i = 0; i < coefficients.matrix.cols(); ++i) {
        label_t label = label_of(i);
        if (labels_of_blocks.empty() || !is_equal(labels_of_blocks.back(), label)) {
            labels_of_blocks.push_back(label);
        }
    }

    // If the states are sorted, no two blocks share the same labels
    std::sort(labels_of_blocks.begin(), labels_of_blocks.end());
    for (size_t i = 1; i < labels_of_blocks.size(); ++i) {
        if (is_equal(labels_of_blocks[i - 1], labels_of_blocks[i])) {
            throw std::invalid_argument("The states are not sorted by the requested labels.");
        }
    }
}

template <typename Derived>
Basis<Derived>::Basis(ketvec_t &&kets)
    : kets(std::move(kets)), coefficients{{static_cast<Eigen::Index>(this->kets.size()),
                                           static_cast<Eigen::Index>(this->kets.size())}} {
    if (this->kets.empty()) {
        throw std::invalid_argument("The basis must contain at least one element.");
    }
    state_index_to_quantum_number_f.reserve(this->kets.size());
    state_index_to_quantum_number_m.reserve(this->kets.size());
    state_index_to_parity.reserve(this->kets.size());
    for (const auto &ket : this->kets) {
        real_t f = std::numeric_limits<real_t>::max();
        real_t m = std::numeric_limits<real_t>::max();
        Parity p = Parity::UNKNOWN;
        // TODO: this is a workaround, and should be fixed, once we restructure the quantum number
        // handling of the Basis class
        if constexpr (requires { ket->get_quantum_number(std::string{}); }) {
            f = ket->get_quantum_number("f");
            m = ket->get_quantum_number("m");
            p = static_cast<Parity>(static_cast<int>(ket->get_quantum_number("parity")));
        } else {
            m = ket->get_quantum_number_m();
        }
        state_index_to_quantum_number_f.push_back(f);
        state_index_to_quantum_number_m.push_back(m);
        state_index_to_parity.push_back(p);
        if (f == std::numeric_limits<real_t>::max()) {
            _has_quantum_number_f = false;
        }
        if (m == std::numeric_limits<real_t>::max()) {
            _has_quantum_number_m = false;
        }
        if (p == Parity::UNKNOWN) {
            _has_parity = false;
        }
    }
    coefficients.matrix.setIdentity();
}

template <typename Derived>
bool Basis<Derived>::has_quantum_number_f() const {
    return _has_quantum_number_f;
}

template <typename Derived>
bool Basis<Derived>::has_quantum_number_m() const {
    return _has_quantum_number_m;
}

template <typename Derived>
bool Basis<Derived>::has_parity() const {
    return _has_parity;
}

template <typename Derived>
const Derived &Basis<Derived>::derived() const {
    return static_cast<const Derived &>(*this);
}

template <typename Derived>
const typename Basis<Derived>::ketvec_t &Basis<Derived>::get_kets() const {
    return kets;
}

template <typename Derived>
const Eigen::SparseMatrix<typename Basis<Derived>::scalar_t, Eigen::RowMajor> &
Basis<Derived>::get_coefficients() const {
    return coefficients.matrix;
}

template <typename Derived>
std::shared_ptr<const Derived> Basis<Derived>::copy_with_coefficients(
    const Eigen::SparseMatrix<scalar_t, Eigen::RowMajor> &values) const {
    if (values.rows() != coefficients.matrix.rows()) {
        throw std::invalid_argument("Incompatible number of rows.");
    }
    if (values.cols() != coefficients.matrix.cols()) {
        throw std::invalid_argument("Incompatible number of columns.");
    }

    // Create a copy of the current object and update the coefficients of the copy
    auto result = std::make_shared<Derived>(derived());

    result->coefficients.matrix = values;

    std::fill(result->state_index_to_quantum_number_f.begin(),
              result->state_index_to_quantum_number_f.end(), std::numeric_limits<real_t>::max());
    std::fill(result->state_index_to_quantum_number_m.begin(),
              result->state_index_to_quantum_number_m.end(), std::numeric_limits<real_t>::max());
    std::fill(result->state_index_to_parity.begin(), result->state_index_to_parity.end(),
              Parity::UNKNOWN);
    result->_has_quantum_number_f = false;
    result->_has_quantum_number_m = false;
    result->_has_parity = false;

    return result;
}

template <typename Derived>
typename Basis<Derived>::real_t Basis<Derived>::get_quantum_number_f(size_t state_index) const {
    real_t quantum_number_f = state_index_to_quantum_number_f.at(state_index);
    if (quantum_number_f == std::numeric_limits<real_t>::max()) {
        throw std::invalid_argument("The state does not have a well-defined quantum number f.");
    }
    return quantum_number_f;
}

template <typename Derived>
typename Basis<Derived>::real_t Basis<Derived>::get_quantum_number_m(size_t state_index) const {
    real_t quantum_number_m = state_index_to_quantum_number_m.at(state_index);
    if (quantum_number_m == std::numeric_limits<real_t>::max()) {
        throw std::invalid_argument("The state does not have a well-defined quantum number m.");
    }
    return quantum_number_m;
}

template <typename Derived>
Parity Basis<Derived>::get_parity(size_t state_index) const {
    Parity parity = state_index_to_parity.at(state_index);
    if (parity == Parity::UNKNOWN) {
        throw std::invalid_argument("The state does not have a well-defined parity.");
    }
    return parity;
}

template <typename Derived>
std::shared_ptr<const Derived> Basis<Derived>::get_state(size_t state_index) const {
    // Create a copy of the current object
    auto restricted = std::make_shared<Derived>(derived());

    // Restrict the copy to the single requested state
    restricted->coefficients.matrix = restricted->coefficients.matrix.col(state_index);

    restricted->state_index_to_quantum_number_f = {state_index_to_quantum_number_f[state_index]};
    restricted->state_index_to_quantum_number_m = {state_index_to_quantum_number_m[state_index]};
    restricted->state_index_to_parity = {state_index_to_parity[state_index]};

    restricted->_has_quantum_number_f =
        restricted->state_index_to_quantum_number_f[0] != std::numeric_limits<real_t>::max();
    restricted->_has_quantum_number_m =
        restricted->state_index_to_quantum_number_m[0] != std::numeric_limits<real_t>::max();
    restricted->_has_parity = restricted->state_index_to_parity[0] != Parity::UNKNOWN;

    return restricted;
}

template <typename Derived>
std::shared_ptr<const typename Basis<Derived>::ket_t>
Basis<Derived>::get_ket(size_t ket_index) const {
    return kets[ket_index];
}

template <typename Derived>
typename Basis<Derived>::Iterator Basis<Derived>::begin() const {
    return kets.begin();
}

template <typename Derived>
typename Basis<Derived>::Iterator Basis<Derived>::end() const {
    return kets.end();
}

template <typename Derived>
Basis<Derived>::Iterator::Iterator(typename ketvec_t::const_iterator it) : it{std::move(it)} {}

template <typename Derived>
bool Basis<Derived>::Iterator::operator!=(const Iterator &other) const {
    return other.it != it;
}

template <typename Derived>
std::shared_ptr<const typename Basis<Derived>::ket_t> Basis<Derived>::Iterator::operator*() const {
    return *it;
}

template <typename Derived>
typename Basis<Derived>::Iterator &Basis<Derived>::Iterator::operator++() {
    ++it;
    return *this;
}

template <typename Derived>
size_t Basis<Derived>::get_number_of_states() const {
    return coefficients.matrix.cols();
}

template <typename Derived>
size_t Basis<Derived>::get_number_of_kets() const {
    return coefficients.matrix.rows();
}

template <typename Derived>
const Transformation<typename Basis<Derived>::scalar_t> &
Basis<Derived>::get_transformation() const {
    return coefficients;
}

template <typename Derived>
Sorting Basis<Derived>::get_sorter(const std::vector<TransformationType> &labels) const {
    // Throw a meaningful error if sorting by energy is requested as this might be a common mistake
    if (std::find(labels.begin(), labels.end(), TransformationType::SORT_BY_ENERGY) !=
        labels.end()) {
        throw std::invalid_argument("States do not store the energy and thus can not be sorted by "
                                    "the energy. Use an energy operator instead.");
    }

    // Initialize transformation
    Sorting transformation;
    transformation.matrix.resize(coefficients.matrix.cols());
    transformation.matrix.setIdentity();

    // Get the sorter
    get_sorter_without_checks(labels, transformation);

    return transformation;
}

template <typename Derived>
std::vector<IndicesOfBlock>
Basis<Derived>::get_indices_of_blocks(const std::vector<TransformationType> &labels) const {
    std::set<TransformationType> unique_labels(labels.begin(), labels.end());
    perform_blocks_checks(unique_labels);

    // Get the blocks
    IndicesOfBlocksCreator blocks_creator({0, static_cast<size_t>(coefficients.matrix.cols())});
    get_indices_of_blocks_without_checks(unique_labels, blocks_creator);

    return blocks_creator.create();
}

template <typename Derived>
void Basis<Derived>::get_sorter_without_checks(const std::vector<TransformationType> &labels,
                                               Sorting &transformation) const {
    constexpr real_t numerical_precision = 100 * std::numeric_limits<real_t>::epsilon();

    int *perm_begin = transformation.matrix.indices().data();
    int *perm_end = perm_begin + coefficients.matrix.cols();
    const int *perm_back = perm_end - 1;

    // Sort the vector based on the requested labels
    set_task_status("Sorting basis states...");
    std::stable_sort(perm_begin, perm_end, [&](int a, int b) {
        for (const auto &label : labels) {
            switch (label) {
            case TransformationType::SORT_BY_PARITY:
                if (state_index_to_parity[a] != state_index_to_parity[b]) {
                    return state_index_to_parity[a] < state_index_to_parity[b];
                }
                break;
            case TransformationType::SORT_BY_QUANTUM_NUMBER_M:
                if (std::abs(state_index_to_quantum_number_m[a] -
                             state_index_to_quantum_number_m[b]) > numerical_precision) {
                    return state_index_to_quantum_number_m[a] < state_index_to_quantum_number_m[b];
                }
                break;
            case TransformationType::SORT_BY_QUANTUM_NUMBER_F:
                if (std::abs(state_index_to_quantum_number_f[a] -
                             state_index_to_quantum_number_f[b]) > numerical_precision) {
                    return state_index_to_quantum_number_f[a] < state_index_to_quantum_number_f[b];
                }
                break;
            default:
                std::abort(); // Can't happen because of previous checks
            }
        }
        return false; // Elements are equal
    });

    // Check for invalid values
    for (const auto &label : labels) {
        switch (label) {
        case TransformationType::SORT_BY_PARITY:
            if (state_index_to_parity[*perm_back] == Parity::UNKNOWN) {
                throw std::invalid_argument(
                    "States cannot be labeled and thus not sorted by the parity.");
            }
            break;
        case TransformationType::SORT_BY_QUANTUM_NUMBER_M:
            if (state_index_to_quantum_number_m[*perm_back] == std::numeric_limits<real_t>::max()) {
                throw std::invalid_argument(
                    "States cannot be labeled and thus not sorted by the quantum number m.");
            }
            break;
        case TransformationType::SORT_BY_QUANTUM_NUMBER_F:
            if (state_index_to_quantum_number_f[*perm_back] == std::numeric_limits<real_t>::max()) {
                throw std::invalid_argument(
                    "States cannot be labeled and thus not sorted by the quantum number f.");
            }
            break;
        default:
            std::abort(); // Can't happen because of previous checks
        }
    }
}

template <typename Derived>
void Basis<Derived>::get_indices_of_blocks_without_checks(
    const std::set<TransformationType> &unique_labels,
    IndicesOfBlocksCreator &blocks_creator) const {
    constexpr real_t numerical_precision = 100 * std::numeric_limits<real_t>::epsilon();

    auto last_quantum_number_f = state_index_to_quantum_number_f[0];
    auto last_quantum_number_m = state_index_to_quantum_number_m[0];
    auto last_parity = state_index_to_parity[0];

    set_task_status("Identifying basis blocks...");
    for (int i = 0; i < coefficients.matrix.cols(); ++i) {
        for (auto label : unique_labels) {
            if (label == TransformationType::SORT_BY_QUANTUM_NUMBER_F &&
                std::abs(state_index_to_quantum_number_f[i] - last_quantum_number_f) >
                    numerical_precision) {
                blocks_creator.add(i);
                break;
            }
            if (label == TransformationType::SORT_BY_QUANTUM_NUMBER_M &&
                std::abs(state_index_to_quantum_number_m[i] - last_quantum_number_m) >
                    numerical_precision) {
                blocks_creator.add(i);
                break;
            }
            if (label == TransformationType::SORT_BY_PARITY &&
                state_index_to_parity[i] != last_parity) {
                blocks_creator.add(i);
                break;
            }
        }
        last_quantum_number_f = state_index_to_quantum_number_f[i];
        last_quantum_number_m = state_index_to_quantum_number_m[i];
        last_parity = state_index_to_parity[i];
    }
}

template <typename Derived>
std::shared_ptr<const Derived> Basis<Derived>::canonicalized() const {
    auto result = std::make_shared<Derived>(derived());

    size_t n = kets.size();

    result->coefficients.matrix.resize(n, n);
    result->coefficients.matrix.setIdentity();

    result->state_index_to_quantum_number_f.resize(n);
    result->state_index_to_quantum_number_m.resize(n);
    result->state_index_to_parity.resize(n);
    result->_has_quantum_number_f = true;
    result->_has_quantum_number_m = true;
    result->_has_parity = true;

    for (size_t i = 0; i < n; ++i) {
        real_t f = std::numeric_limits<real_t>::max();
        real_t m = std::numeric_limits<real_t>::max();
        Parity p = Parity::UNKNOWN;
        // TODO: this is a workaround, and should be fixed, once we restructure the quantum number
        // handling of the Basis class
        if constexpr (requires { kets[i]->get_quantum_number(std::string{}); }) {
            f = kets[i]->get_quantum_number("f");
            m = kets[i]->get_quantum_number("m");
            p = static_cast<Parity>(static_cast<int>(kets[i]->get_quantum_number("parity")));
        } else {
            m = kets[i]->get_quantum_number_m();
        }
        result->state_index_to_quantum_number_f[i] = f;
        result->state_index_to_quantum_number_m[i] = m;
        result->state_index_to_parity[i] = p;
        if (f == std::numeric_limits<real_t>::max()) {
            result->_has_quantum_number_f = false;
        }
        if (m == std::numeric_limits<real_t>::max()) {
            result->_has_quantum_number_m = false;
        }
        if (p == Parity::UNKNOWN) {
            result->_has_parity = false;
        }
    }

    return result;
}

template <typename Derived>
bool Basis<Derived>::is_canonical() const {
    constexpr real_t numerical_precision = 100 * std::numeric_limits<real_t>::epsilon();

    // The basis is canonical if and only if its coefficient matrix is the identity matrix, i.e.,
    // if the i-th state is the i-th ket
    if (coefficients.matrix.rows() != coefficients.matrix.cols()) {
        return false;
    }

    Eigen::Index num_ones = 0;
    for (Eigen::Index row = 0; row < coefficients.matrix.outerSize(); ++row) {
        for (typename Eigen::SparseMatrix<scalar_t, Eigen::RowMajor>::InnerIterator it(
                 coefficients.matrix, row);
             it; ++it) {
            if (it.row() == it.col() &&
                std::abs(it.value() - static_cast<scalar_t>(1)) <= numerical_precision) {
                ++num_ones;
            } else if (std::abs(it.value()) > numerical_precision) {
                return false;
            }
        }
    }
    return num_ones == coefficients.matrix.rows();
}

template <typename Derived>
std::shared_ptr<const Derived> Basis<Derived>::transformed(const Sorting &transformation) const {
    // Create a copy of the current object
    auto transformed = std::make_shared<Derived>(derived());

    if (coefficients.matrix.cols() == 0) {
        return transformed;
    }

    // Apply the transformation
    set_task_status("Applying basis sorting...");
    transformed->coefficients.matrix = coefficients.matrix * transformation.matrix;

    transformed->state_index_to_quantum_number_f.resize(transformation.matrix.size());
    transformed->state_index_to_quantum_number_m.resize(transformation.matrix.size());
    transformed->state_index_to_parity.resize(transformation.matrix.size());

    set_task_status("Relabeling sorted basis states...");
    for (int i = 0; i < transformation.matrix.size(); ++i) {
        transformed->state_index_to_quantum_number_f[i] =
            state_index_to_quantum_number_f[transformation.matrix.indices()[i]];
        transformed->state_index_to_quantum_number_m[i] =
            state_index_to_quantum_number_m[transformation.matrix.indices()[i]];
        transformed->state_index_to_parity[i] =
            state_index_to_parity[transformation.matrix.indices()[i]];
    }

    return transformed;
}

template <typename Derived>
std::shared_ptr<const Derived>
Basis<Derived>::transformed(const Transformation<scalar_t> &transformation) const {
    // TODO why is "numerical_precision = 100 * std::sqrt(coefficients.matrix.rows()) *
    // std::numeric_limits<real_t>::epsilon()" too small for figuring out whether m is conserved?
    real_t numerical_precision = 0.001;

    // Create a copy of the current object
    auto transformed = std::make_shared<Derived>(derived());

    if (coefficients.matrix.cols() == 0) {
        return transformed;
    }

    // Apply the transformation
    // If a quantum number turns out to be conserved by the transformation, it will be
    // rounded to the nearest half integer to avoid loss of numerical_precision.
    set_task_status("Applying basis transformation...");
    transformed->coefficients.matrix = coefficients.matrix * transformation.matrix;

    Eigen::SparseMatrix<real_t> probs = transformation.matrix.cwiseAbs2().transpose();

    set_task_status("Updating transformed quantum numbers...");
    {
        auto map = Eigen::Map<const Eigen::VectorX<real_t>>(state_index_to_quantum_number_f.data(),
                                                            state_index_to_quantum_number_f.size());
        Eigen::VectorX<real_t> val = probs * map;
        Eigen::VectorX<real_t> sq = probs * map.cwiseAbs2();
        Eigen::VectorX<real_t> diff = (val.cwiseAbs2() - sq).cwiseAbs();
        transformed->state_index_to_quantum_number_f.resize(probs.rows());

        for (size_t i = 0; i < transformed->state_index_to_quantum_number_f.size(); ++i) {
            if (diff[i] < numerical_precision) {
                transformed->state_index_to_quantum_number_f[i] = std::round(val[i] * 2) / 2;
            } else {
                transformed->state_index_to_quantum_number_f[i] =
                    std::numeric_limits<real_t>::max();
                transformed->_has_quantum_number_f = false;
            }
        }
    }

    {
        auto map = Eigen::Map<const Eigen::VectorX<real_t>>(state_index_to_quantum_number_m.data(),
                                                            state_index_to_quantum_number_m.size());
        Eigen::VectorX<real_t> val = probs * map;
        Eigen::VectorX<real_t> sq = probs * map.cwiseAbs2();
        Eigen::VectorX<real_t> diff = (val.cwiseAbs2() - sq).cwiseAbs();
        transformed->state_index_to_quantum_number_m.resize(probs.rows());

        for (size_t i = 0; i < transformed->state_index_to_quantum_number_m.size(); ++i) {
            if (diff[i] < numerical_precision) {
                transformed->state_index_to_quantum_number_m[i] = std::round(val[i] * 2) / 2;
            } else {
                transformed->state_index_to_quantum_number_m[i] =
                    std::numeric_limits<real_t>::max();
                transformed->_has_quantum_number_m = false;
            }
        }
    }

    {
        using utype = std::underlying_type<Parity>::type;
        Eigen::VectorX<real_t> map(state_index_to_parity.size());
        for (size_t i = 0; i < state_index_to_parity.size(); ++i) {
            map[i] = static_cast<utype>(state_index_to_parity[i]);
        }
        Eigen::VectorX<real_t> val = probs * map;
        Eigen::VectorX<real_t> sq = probs * map.cwiseAbs2();
        Eigen::VectorX<real_t> diff = (val.cwiseAbs2() - sq).cwiseAbs();
        transformed->state_index_to_parity.resize(probs.rows());

        for (size_t i = 0; i < transformed->state_index_to_parity.size(); ++i) {
            if (diff[i] < numerical_precision) {
                transformed->state_index_to_parity[i] = static_cast<Parity>(std::lround(val[i]));
            } else {
                transformed->state_index_to_parity[i] = Parity::UNKNOWN;
                transformed->_has_parity = false;
            }
        }
    }

    return transformed;
}

// Explicit instantiations
template class Basis<BasisAtom<double>>;
template class Basis<BasisAtom<std::complex<double>>>;
template class Basis<BasisPair<double>>;
template class Basis<BasisPair<std::complex<double>>>;
} // namespace pairinteraction
