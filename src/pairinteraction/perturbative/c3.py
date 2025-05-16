# SPDX-FileCopyrightText: 2025 Pairinteraction Developers
# SPDX-License-Identifier: LGPL-3.0-or-later

import logging
from collections.abc import Collection
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
def get_c3_from_system(
    ket_tuple_list: Collection["KetPairLike"], system_pair: "SystemPair", *, unit: None = None
) -> "PintFloat": ...


@overload
def get_c3_from_system(ket_tuple_list: Collection["KetPairLike"], system_pair: "SystemPair", unit: str) -> float: ...


def get_c3_from_system(
    ket_tuple_list: Collection["KetPairLike"], system_pair: "SystemPair", unit: Optional[str] = None
) -> Union[float, "PintFloat"]:
    r"""Calculate the :math:`C_3` coefficient for a list of two 2-tuples of single atom ket states.

    This function calculates the :math:`C_3` coefficient in the desired unit. The input is a list of two 2-tuples of
    single atom ket states. We use the convention :math:`\Delta E = \frac{C_3}{r^3}`.

    Args:
        ket_tuple_list: The input as a list of tuples of two states [(a,b),(c,d)],
            the :math:`C_3` coefficient is calculated for (a,b)->(c,d).
            If there are not exactly two tuples in the list, a ValueError is raised.
        system_pair: The pair system that is used for the calculation.
        unit: The unit to which to convert the result. Default None will return a pint quantity.

    Returns:
        The :math:`C_3` coefficient with its unit.

    Raises:
        ValueError: If a list of not exactly two tuples of single atom states is given.

    """
    if len(ket_tuple_list) != 2:
        raise ValueError("C3 coefficient can be calculated only between two 2-atom states.")

    r = system_pair.get_distance()
    if np.isinf(r.magnitude):
        logger.warning(
            "Pair system is initialized without a distance. "
            "Calculating the C3 coefficient at a distance vector of [0, 0, 20] mum."
        )
        old_distance_vector = system_pair.get_distance_vector()
        system_pair.set_distance_vector([0, 0, 20], "micrometer")
        c3 = get_c3_from_system(ket_tuple_list, system_pair, unit=unit)
        system_pair.set_distance_vector(old_distance_vector)
        return c3

    h_eff, _ = get_effective_hamiltonian_from_system(ket_tuple_list, system_pair, order=1)
    c3_pint = h_eff[0, 1] * r**3  # type: ignore [index] # PintArray does not know it can be indexed
    return QuantityScalar.from_pint(c3_pint, "c3").to_pint_or_unit(unit)
