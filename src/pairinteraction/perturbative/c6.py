# SPDX-FileCopyrightText: 2025 Pairinteraction Developers
# SPDX-License-Identifier: LGPL-3.0-or-later

import logging
from collections.abc import Iterable
from typing import TYPE_CHECKING, Optional, Union, overload

import numpy as np

from pairinteraction.perturbative.effective_hamiltonian import get_effective_hamiltonian_from_system
from pairinteraction.units import QuantityScalar

if TYPE_CHECKING:
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
    from pairinteraction.units import PintFloat

    SystemPair = Union[pi_real.SystemPair, pi_complex.SystemPair]

logger = logging.getLogger(__name__)


@overload
def get_c6_from_system(ket_tuple: "KetPairLike", system_pair: "SystemPair", *, unit: None = None) -> "PintFloat": ...


@overload
def get_c6_from_system(ket_tuple: "KetPairLike", system_pair: "SystemPair", unit: str) -> float: ...


def get_c6_from_system(
    ket_tuple: "KetPairLike", system_pair: "SystemPair", unit: Optional[str] = None
) -> Union[float, "PintFloat"]:
    r"""Calculate the :math:`C_6` coefficient for a given tuple of ket states.

    This function calculates the :math:`C_6` coefficient in the desired unit. The input is a 2-tuple of single atom ket
    states.

    Args:
        ket_tuple: The input is a tuple repeating the same single atom state in the format (a,a).
        If a tuple with not exactly two identical states is given, a ValueError is raised.
        system_pair: The pair system that is used for the calculation.
        unit: The unit to which to convert the result. Default None will return a pint quantity.

    Returns:
        The :math:`C_6` coefficient. If a unit is specified, the value in this unit is returned.

    Raises:
        ValueError: If a tuple with more than two single atom states is given.

    """
    if isinstance(ket_tuple, Iterable):
        if len(ket_tuple) != 2:
            raise ValueError("C6 coefficient can be calculated only for a single 2-atom state.")
        if ket_tuple[0].species == ket_tuple[1].species and ket_tuple[0] != ket_tuple[1]:
            raise ValueError(
                "If you want to calculate 2nd order perturbations of two different states a and b, "
                "please use the get_effective_hamiltonian_from_system([(a,b), (b,a)], system_pair) function."
            )

    r = system_pair.get_distance()
    if np.isinf(r.magnitude):
        logger.warning(
            "Pair system is initialized without a distance. "
            "Calculating the C6 coefficient at a distance vector of [0, 0, 20] mum."
        )
        old_distance_vector = system_pair.get_distance_vector()
        system_pair.set_distance_vector([0, 0, 20], "micrometer")
        c6 = get_c6_from_system(ket_tuple, system_pair, unit=unit)
        system_pair.set_distance_vector(old_distance_vector)
        return c6

    h_eff, _ = get_effective_hamiltonian_from_system([ket_tuple], system_pair, order=2)
    h_0, _ = get_effective_hamiltonian_from_system([ket_tuple], system_pair, order=0)
    c6_pint = (h_eff[0, 0] - h_0[0, 0]) * r**6  # type: ignore [index] # PintArray does not know it can be indexed
    return QuantityScalar.from_pint(c6_pint, "c6").to_pint_or_unit(unit)
