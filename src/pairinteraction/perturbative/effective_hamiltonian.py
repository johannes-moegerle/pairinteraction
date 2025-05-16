# SPDX-FileCopyrightText: 2025 Pairinteraction Developers
# SPDX-License-Identifier: LGPL-3.0-or-later

import logging
from collections.abc import Collection
from typing import TYPE_CHECKING, Optional, Union, overload

import numpy as np
from scipy import sparse

from pairinteraction.units import QuantityArray

if TYPE_CHECKING:
    from scipy.sparse import csr_matrix

    from pairinteraction import (
        complex as pi_complex,
        real as pi_real,
    )
    from pairinteraction._wrapped.ket.ket_atom import KetAtom  # noqa: F401  # required for sphinx for KetPairLike
    from pairinteraction._wrapped.ket.ket_pair import (
        KetPairComplex,  # noqa: F401  # required for sphinx for KetPairLike
        KetPairLike,
        KetPairReal,  # noqa: F401  # required for sphinx for KetPairLike
    )
    from pairinteraction.units import NDArray, PintArray

    SystemPair = Union[pi_real.SystemPair, pi_complex.SystemPair]

logger = logging.getLogger(__name__)


@overload
def get_effective_hamiltonian_from_system(
    ket_tuple_list: Collection["KetPairLike"],
    system_pair: "SystemPair",
    order: int = 2,
    required_overlap: float = 0.9,
    *,
    unit: None = None,
) -> tuple["PintArray", "csr_matrix"]: ...


@overload
def get_effective_hamiltonian_from_system(
    ket_tuple_list: Collection["KetPairLike"],
    system_pair: "SystemPair",
    order: int = 2,
    required_overlap: float = 0.9,
    *,
    unit: str,
) -> tuple["NDArray", "csr_matrix"]: ...


def get_effective_hamiltonian_from_system(
    ket_tuple_list: Collection["KetPairLike"],
    system_pair: "SystemPair",
    order: int = 2,
    required_overlap: float = 0.9,
    unit: Optional[str] = None,
) -> tuple[Union["NDArray", "PintArray"], "csr_matrix"]:
    r"""Get the perturbative Hamiltonian at a desired order in Rayleigh-Schrödinger perturbation theory.

    This function takes a list of tuples of ket states, which forms the basis of the model space in which the effective
    Hamiltonian is calculated. The whole Hamiltonian is taken from a pair system.
    The Hamiltonian of the pair system is assumed to be diagonal in the unperturbed Hamiltonian, all off-diagonal
    elements are assumed to belong to the perturbative term.
    The function also checks for resonances between all states and states in the model space.

    Args:
        ket_tuple_list: List of all pair states that span up the model space.
            The effective Hamiltonian is calculated for these states.
        system_pair: Two-Atom-System, diagonal in the basis of the unperturbed Hamiltonian.
        order: Order up to which the perturbation theory is expanded. Support up to third order.
            Default is second order.
        required_overlap: If set, the code checks for validity of a perturbative treatment.
            Error is thrown if the perturbed eigenstate has less overlap than this value with the
            unperturbed eigenstate.
        unit: The unit to which to convert the result. Default None will return a pint quantity.

    Returns:
        - Effective Hamiltonian as a :math:`m \times m` matrix, where m is the length of `ket_tuple_list`.
        - Eigenvectors in perturbation theory due to interaction with states out of the model space, returned as
          a sparse matrix in compressed row format. Each row represents the corresponding eigenvector.

    Raises:
        ValueError: If a resonance between a state in the model space and a state not in the model space occurs.

    """
    if np.isinf(system_pair.get_distance().magnitude):
        raise ValueError(
            "Pair system is initialized without a distance. "
            "Please set a distance for calculating an effective Hamiltonian."
        )

    model_inds = _get_model_inds(ket_tuple_list, system_pair)
    h_au = system_pair.get_hamiltonian().to_base_units().magnitude  # Hamiltonian in atomic units
    h_eff_au, eigvec_perturb = _calculate_perturbative_hamiltonian(h_au, model_inds, order)
    if not 0 <= required_overlap <= 1:
        raise ValueError("Required overlap has to be a positive real number between zero and one.")
    if required_overlap > 0:
        _check_for_resonances(model_inds, eigvec_perturb, system_pair, required_overlap)

    h_eff = QuantityArray.from_base_unit(h_eff_au, "energy").to_pint_or_unit(unit)
    return h_eff, eigvec_perturb


def _calculate_perturbative_hamiltonian(
    hamiltonian: "csr_matrix", model_inds: list[int], order: int = 2
) -> tuple["NDArray", "csr_matrix"]:
    r"""Calculate the perturbative Hamiltonian at a given order.

    This function takes a Hamiltonian as a sparse matrix which is diagonal in the unperturbed basis
    and list of indices spanning up the model space.
    It calculates both the effective Hamiltonian, spanned up by the states of the model space, as well as the
    perturbed eigenstates due to interactions with the exterior space in the desired order of perturbation theory.

    Args:
        hamiltonian: Quadratic hermitian matrix. Perturbative terms are assumed to be only off-diagonal.
        model_inds: List of indices corresponding to the states that span up the model space.
        order: Order up to which the perturbation theory is expanded. Support up to third order.
            Default is second order.

    Returns:
        Effective Hamiltonian as a :math:`m \times m` matrix, where m is the length of `ket_tuple_list`
        Eigenvectors in perturbation theory due to interaction with states out of the model
            space, returned as a sparse matrix in compressed row format. Each row represent the
            corresponding eigenvector

    """
    m_inds = np.array(model_inds)
    o_inds = np.setdiff1d(np.arange(hamiltonian.shape[0]), m_inds)
    h_eff, eigvec_perturb = _calculate_unsorted_perturbative_hamiltonian(hamiltonian, m_inds, o_inds, order)

    # resort eigvec to original order
    all_inds = np.append(m_inds, o_inds)
    all_inds_positions = np.argsort(all_inds)
    eigvec_perturb = eigvec_perturb[:, all_inds_positions]

    # include the hermitian conjugate part of the effective Hamiltonian
    h_eff = 0.5 * (h_eff + h_eff.conj().T)

    return h_eff, eigvec_perturb


def _calculate_unsorted_perturbative_hamiltonian(
    hamiltonian: "csr_matrix", m_inds: "NDArray", o_inds: "NDArray", order: int
) -> tuple["NDArray", "csr_matrix"]:
    # This function is outsourced from _calculate_perturbative_hamiltonian to allow for better type checking
    if order not in [0, 1, 2, 3]:
        raise ValueError("Perturbation theory is only implemented for orders [0, 1, 2, 3].")

    h0 = hamiltonian.diagonal()
    h0_m = h0[m_inds]

    h_eff = np.diag(h0_m)
    eigvec_perturb = sparse.hstack(
        [sparse.eye(len(m_inds), len(m_inds), format="csr").tocsr(), sparse.csr_matrix((len(m_inds), len(o_inds)))]
    ).tocsr()

    if order < 1:
        return h_eff, eigvec_perturb

    v_offdiag = hamiltonian - sparse.diags(h0)
    v_mm = v_offdiag[np.ix_(m_inds, m_inds)]
    h_eff += v_mm

    if order < 2:
        return h_eff, eigvec_perturb

    h0_e = h0[o_inds]
    v_me = v_offdiag[np.ix_(m_inds, o_inds)]
    delta_e_em = 1 / (h0_m[np.newaxis, :] - h0_e[:, np.newaxis])
    h_eff += v_me @ ((v_me.conj().T).multiply(delta_e_em))
    addition_mm = sparse.csr_matrix((len(m_inds), len(m_inds)))
    addition_me = sparse.csr_matrix(((v_me.conj().T).multiply(delta_e_em)).T)
    eigvec_perturb = eigvec_perturb + sparse.hstack([addition_mm, addition_me])

    if order < 3:
        return h_eff, eigvec_perturb

    diff = h0_m[np.newaxis, :] - h0_m[:, np.newaxis]
    diff = np.where(diff == 0, np.inf, diff)
    delta_e_mm = 1 / diff
    v_ee = v_offdiag[np.ix_(o_inds, o_inds)]
    if len(m_inds) > 1:
        logger.warning(
            "At third order, the eigenstates are currently only valid when only one state is in the model space. "
            "Take care with interpreation of the perturbed eigenvectors."
        )
    h_eff += v_me @ (
        (v_ee @ ((v_me.conj().T).multiply(delta_e_em)) - ((v_me.conj().T).multiply(delta_e_em)) @ v_mm).multiply(
            delta_e_em
        )
    )
    addition_mm_diag = -0.5 * sparse.csr_matrix(
        sparse.diags((v_me @ ((v_me.conj().T).multiply(np.square(delta_e_em)))).diagonal())
    )
    addition_mm_offdiag = sparse.csr_matrix(((v_me @ (v_me.conj().T).multiply(delta_e_em)).multiply(delta_e_mm)).T)
    addition_me = sparse.csr_matrix(((v_ee @ ((v_me.conj().T).multiply(delta_e_em))).multiply(delta_e_em)).T)
    addition_me_2 = sparse.csr_matrix(((v_me.conj().T @ ((v_mm.conj().T).multiply(delta_e_mm))).multiply(delta_e_em)).T)
    eigvec_perturb = eigvec_perturb + sparse.hstack(
        [addition_mm_diag + addition_mm_offdiag, addition_me + addition_me_2]
    )

    return h_eff, eigvec_perturb


def _get_model_inds(ket_tuple_list: Collection["KetPairLike"], system_pair: "SystemPair") -> list[int]:
    """Get the indices of all ket tuples in the basis of pair system.

    This function takes a list of 2-tuples of ket states, and a pair system holding the entire basis.
    It returns an array of indices of the pair system basis in the order of the tuple list.

    Args:
        ket_tuple_list: List of all pair states that span up the model space.
        system_pair: Two-Atom-System, diagonal in the basis of the unperturbed Hamiltonian.

    Returns:
        List of indices corresponding to the states that span up the model space.

    """
    model_inds = []
    for kets in ket_tuple_list:
        overlap = system_pair.basis.get_overlaps(kets)
        index = np.argmax(overlap)
        if overlap[index] == 0:
            raise ValueError(f"The pairstate {kets} is not part of the basis of the pair system.")
        if overlap[index] < 0.5:
            raise ValueError(f"The pairstate {kets} cannot be identified uniquely (max overlap: {overlap[index]}).")
        model_inds.append(int(index))
    return model_inds


def _check_for_resonances(
    model_inds: list[int],
    eigvec_perturb: "csr_matrix",
    system_pair: "SystemPair",
    required_overlap: float,
) -> None:
    r"""Check for resonance between the states in the model space and other states.

    This function takes the perturbed eigenvectors of the perturbation theory as an input.
    If the overlap of the perturbed eigenstate with its corresponding unperturbed state are too small,
    this function raises an error, as perturbation theory breaks down.
    In this case, it also prints all states with a relevant admixture that should therefore be also included in the
    model space, to allow perturbation theory.

    Args:
        model_inds: List of indices corresponding to the states that span up the model space.
        eigvec_perturb: Sparse representation of the perturbed eigenstates in the desired order of
            perturbation theory. Each row corresponds to the eigestate according to `state model indices.`
        system_pair: Two-Atom-System, diagonal in the basis of the unperturbed Hamiltonian.
        order: Order up to which the perturbation theory is expanded. Support up to third order.
            Default is second order.
        required_overlap: If set, the code checks for validity of a perturbative treatment.
            Error is thrown if the perturbed eigenstate has less overlap than this value with the unperturbed eigenstate

    Returns:
        Effective Hamiltonian as a :math:`m \times m` matrix, where m is the length of `ket_tuple_list`
        Eigenvectors in perturbation theory due to interaction with states out of the model space,
            returned as a sparse matrix in compressed row format. Each row represent the corresponding eigenvector

    Raises:
        ValueError: If a resonance between a state in the model space and a state not in the model space occurs.

    """
    overlaps = (eigvec_perturb.multiply(eigvec_perturb.conj())).real
    error_flag = False
    for i, j in zip(range(len(model_inds)), model_inds):
        vector_norm = sparse.linalg.norm(overlaps[i, :])
        overlap = overlaps[i, j] / vector_norm
        if overlap >= required_overlap:
            continue
        error_flag = True
        print_above_admixture = (1 - required_overlap) * 0.05
        indices = sparse.find(overlaps[i, :] >= print_above_admixture * vector_norm)[1]
        logger.error(
            "The state %s has resonances with the following states, please consider adding them to your model space:",
            system_pair.basis.kets[j],
        )
        for index in indices:
            if index == j:
                continue
            admixture = 1 if np.isinf(overlaps[i, index]) else overlaps[i, index] / vector_norm
            logger.error("  - %s with admixture %.3f", system_pair.basis.kets[index], admixture)
    if error_flag:
        raise ValueError(
            "Error. Perturbative Calculation not possible due to resonances. "
            "Add more states to the model space or adapt your required overlap."
        )
