# SPDX-FileCopyrightText: 2025 Pairinteraction Developers
# SPDX-License-Identifier: LGPL-3.0-or-later

from collections.abc import Collection
from typing import TYPE_CHECKING, Any, Optional, Union, overload

from pairinteraction.perturbative.effective_hamiltonian import get_effective_hamiltonian_from_system
from pairinteraction.units import QuantityScalar

if TYPE_CHECKING:
    from pairinteraction._wrapped.ket.ket_pair import KetAtomTuple
    from pairinteraction._wrapped.system.system_pair import SystemPair
    from pairinteraction.units import PintFloat


@overload
def get_c3_from_system(
    ket_tuples: Collection["KetAtomTuple"], system_pair: "SystemPair[Any]", unit: None = None
) -> "PintFloat": ...


@overload
def get_c3_from_system(ket_tuples: Collection["KetAtomTuple"], system_pair: "SystemPair[Any]", unit: str) -> float: ...


def get_c3_from_system(
    ket_tuples: Collection["KetAtomTuple"], system_pair: "SystemPair[Any]", unit: Optional[str] = None
) -> Union[float, "PintFloat"]:
    r"""Calculate the :math:`C_3` coefficient between two pair states.

    This function takes a list of two KetAtom 2-tuples (i.e. [(ketA_1, ketB_1), (ketA_2, ketB_2)]),
    and calculates the :math:`C_3` coefficient for the coupling (ketA_1, ketB_1) <-> (ketA_2, ketB_2).

    We use the convention :math:`\Delta E = \frac{C_3}{r^3}`.

    Args:
        ket_tuples: List of two KetAtom 2-tuples: [(ketA_1, ketB_1), (ketA_2, ketB_2)])
            between which the :math:`C_3` coefficient is calculated.
        system_pair: SystemPair object that defines the Hamiltonian
            (including electric and magnetic fields, the interatomic distance, ...)
        unit: The unit in which the :math:`C_3` coefficient will be returned.
            Default None will return a pint quantity.

    Returns:
        The :math:`C_3` coefficient.

    """
    if len(ket_tuples) != 2:
        raise ValueError(
            "The C3 coefficient can only be calculated between two pair states."
            " Thus only provide two KetAtoms 2-tuples as ket_tuples argument."
        )
    h_eff, _ = get_effective_hamiltonian_from_system(
        ket_tuples, system_pair, perturbation_order=1, return_only_specified_order=True
    )
    c3_pint = h_eff[0, 1] * system_pair.get_distance() ** 3  # type: ignore [index] # PintArray does not know it can be indexed
    return QuantityScalar.from_pint(c3_pint, "c3").to_pint_or_unit(unit)
