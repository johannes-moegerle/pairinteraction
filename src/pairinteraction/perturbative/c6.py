# SPDX-FileCopyrightText: 2025 Pairinteraction Developers
# SPDX-License-Identifier: LGPL-3.0-or-later

from typing import TYPE_CHECKING, Any, Optional, Union, overload

from pairinteraction._wrapped.ket.ket_atom import KetAtom
from pairinteraction.perturbative.effective_hamiltonian import get_effective_hamiltonian_from_system
from pairinteraction.units import QuantityScalar

if TYPE_CHECKING:
    from pairinteraction._wrapped.ket.ket_pair import KetAtomTuple
    from pairinteraction._wrapped.system.system_pair import SystemPair
    from pairinteraction.units import PintFloat


@overload
def get_c6_from_system(ket_tuple: "KetAtomTuple", system_pair: "SystemPair[Any]", unit: None = None) -> "PintFloat": ...


@overload
def get_c6_from_system(ket_tuple: "KetAtomTuple", system_pair: "SystemPair[Any]", unit: str) -> float: ...


def get_c6_from_system(
    ket_tuple: "KetAtomTuple", system_pair: "SystemPair[Any]", unit: Optional[str] = None
) -> Union[float, "PintFloat"]:
    r"""Calculate the :math:`C_6` coefficient for a given pair state.

    This function takes a tuple of two KetAtom (i.e. (ketA, ketB) ),
    and calculates the :math:`C_6` coefficient of the corresponding pair state.
    If ketA and ketB are of the same species, they must be identical.
    If you want to calculate the second order perturbation corrections for two different states, of the same species,
    use the `get_effective_hamiltonian_from_system(...)` function instead.

    Args:
        ket_tuple: Tuple of two KetAtom: (ketA, ketB) for which the :math:`C_6` coefficient is calculated.
        system_pair: SystemPair object that defines the Hamiltonian
            (including electric and magnetic fields, the interatomic distance, ...)
        unit: The unit in which the :math:`C_6` coefficient will be returned.
            Default None will return a pint quantity.

    Returns:
        The :math:`C_6` coefficient.

    """
    if len(ket_tuple) != 2 or not isinstance(ket_tuple[0], KetAtom):
        raise ValueError("The C6 coefficient can only be calculated for a tuple of two KetAtoms as ket_tuple argument.")
    if ket_tuple[0].species == ket_tuple[1].species and ket_tuple[0] != ket_tuple[1]:
        raise ValueError(
            "If you want to calculate second order perturbation corrections for two different states "
            "use the `get_effective_hamiltonian_from_system(...)` function."
        )
    h_eff, _ = get_effective_hamiltonian_from_system(
        [ket_tuple], system_pair, perturbation_order=2, return_only_specified_order=True
    )
    c6_pint = h_eff[0, 0] * system_pair.get_distance() ** 6  # type: ignore [index] # PintArray does not know it can be indexed
    return QuantityScalar.from_pint(c6_pint, "c6").to_pint_or_unit(unit)
