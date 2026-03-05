# SPDX-FileCopyrightText: 2024 PairInteraction Developers
# SPDX-License-Identifier: LGPL-3.0-or-later
from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal, Union

from typing_extensions import TypeGuard

from pairinteraction.ket.ket_atom import KetAtom
from pairinteraction.ket.ket_base import KetBase

if TYPE_CHECKING:
    from collections.abc import Collection

    from pairinteraction import _backend
    from pairinteraction.state import StateAtom
    from pairinteraction.system.system_atom import SystemAtom


KetAtomTuple = Union[tuple["KetAtom", "KetAtom"], Sequence["KetAtom"]]
KetPairLike = Union["KetPair", KetAtomTuple]


def is_ket_pair_like(obj: Any) -> TypeGuard[KetPairLike]:
    return isinstance(obj, KetPair) or is_ket_atom_tuple(obj)


def is_ket_atom_tuple(obj: Any) -> TypeGuard[KetAtomTuple]:
    return hasattr(obj, "__len__") and len(obj) == 2 and all(isinstance(x, KetAtom) for x in obj)


class KetPair(KetBase):
    """Ket for a pair state of two atoms.

    For pair systems, we choose KetPair object as the product states of the single-atom eigenstates.
    Thus, the Ket pair objects depend on the system and the applied fields.
    Therefore for different pair systems the KetPair objects are not necessarily orthogonal anymore.

    A KetPair object can be created directly from two diagonalized :class:`pairinteraction.SystemAtom` objects
    and two :class:`pairinteraction.KetAtom` objects, or they can be obtained from a
    :class:`pairinteraction.BasisPair` object.

    """

    _cpp: _backend.KetPairComplex

    def __init__(self, systems: Collection[SystemAtom], kets: KetAtomTuple) -> None:
        """Create a KetPair from two diagonalized SystemAtom objects and two KetAtom objects.

        Args:
            systems: A collection of exactly two diagonalized :class:`pairinteraction.SystemAtom` objects.
            kets: A pair of :class:`pairinteraction.KetAtom` objects identifying the desired single-atom states.

        Raises:
            ValueError: If not exactly 2 systems and 2 kets are given, or if a system is not diagonalized.

        """
        import numpy as np

        from pairinteraction import _backend
        from pairinteraction.system.system_atom import SystemAtomReal

        sys_list = list(systems)
        ket_list = list(kets)

        if len(sys_list) != 2 or len(ket_list) != 2:
            raise ValueError("KetPair requires exactly 2 systems and 2 kets.")

        sys1, sys2 = sys_list
        ket1, ket2 = ket_list

        if not sys1.is_diagonal or not sys2.is_diagonal:
            raise ValueError("Both systems must be diagonalized before creating a KetPair.")

        use_real = isinstance(sys1, SystemAtomReal) and isinstance(sys2, SystemAtomReal)

        basis1 = sys1.basis
        basis2 = sys2.basis
        idx1 = basis1.get_corresponding_state_index(ket1)
        idx2 = basis2.get_corresponding_state_index(ket2)

        eigenenergies1 = np.array(sys1._cpp.get_eigenenergies())
        eigenenergies2 = np.array(sys2._cpp.get_eigenenergies())
        energy = float(eigenenergies1[idx1] + eigenenergies2[idx2])

        if use_real:
            self._cpp = _backend.KetPairReal.create(basis1._cpp, basis2._cpp, idx1, idx2, energy)  # type: ignore [arg-type, assignment]
        else:
            self._cpp = _backend.KetPairComplex.create(basis1._cpp, basis2._cpp, idx1, idx2, energy)

    def get_label(self, fmt: Literal["raw", "ket", "bra", "detailed"] = "raw", *, max_kets: int = 3) -> str:
        """Label representing the ket pair.

        Args:
            fmt: The format of the label, i.e. whether to return the raw label, or the label in ket or bra notation.
            max_kets: Maximum number of single atom kets to include in the label for each StateAtom.

        Returns:
            A string representation of the ket pair.

        """
        if fmt == "detailed":
            atom_labels = [atom.get_label(max_kets=max_kets) for atom in self.state_atoms]
            return f"({atom_labels[0]}) ⊗ ({atom_labels[1]})"
        return super().get_label(fmt)

    @property
    def state_atoms(self) -> tuple[StateAtom, StateAtom]:
        """Return the state atoms of the ket pair."""
        from pairinteraction.state import StateAtom, StateAtomReal

        _state_atom_class = StateAtomReal if isinstance(self, KetPairReal) else StateAtom

        state_atoms = []
        for atomic_state in self._cpp.get_atomic_states():
            state = _state_atom_class._from_cpp_object(atomic_state)
            state_atoms.append(state)
        return tuple(state_atoms)  # type: ignore [return-value]


class KetPairReal(KetPair):
    _cpp: _backend.KetPairReal  # type: ignore [assignment]
