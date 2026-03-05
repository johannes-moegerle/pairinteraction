# SPDX-FileCopyrightText: 2024 PairInteraction Developers
# SPDX-License-Identifier: LGPL-3.0-or-later
from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal, Union

from typing_extensions import TypeGuard

from pairinteraction import _backend
from pairinteraction.ket.ket_atom import KetAtom
from pairinteraction.ket.ket_base import KetBase

if TYPE_CHECKING:
    from collections.abc import Collection

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

    A KetPair object can be created directly from two :class:`pairinteraction.KetAtom` objects and
    two diagonalized :class:`pairinteraction.SystemAtom` objects, or they can be obtained from a
    :class:`pairinteraction.BasisPair` object.

    """

    _cpp: _backend.KetPairComplex
    _cpp_type = _backend.KetPairComplex

    def __init__(self, ket_tuple: KetAtomTuple, systems: Collection[SystemAtom]) -> None:
        """Create a KetPair from two diagonalized SystemAtom objects and two KetAtom objects.

        Args:
            ket_tuple: A pair of :class:`pairinteraction.KetAtom` objects identifying the desired single-atom states.
            systems: A collection of exactly two diagonalized :class:`pairinteraction.SystemAtom` objects.

        Raises:
            ValueError: If not exactly 2 systems and 2 kets are given, or if a system is not diagonalized.

        """
        if len(systems) != 2 or len(ket_tuple) != 2:
            raise ValueError("KetPair requires exactly 2 systems and 2 kets.")

        if not all(sys.is_diagonal for sys in systems):
            raise ValueError("Both systems must be diagonalized before creating a KetPair.")

        basis_atoms = [system.basis for system in systems]
        ket_indices = [basis.get_corresponding_state_index(ket) for basis, ket in zip(basis_atoms, ket_tuple)]

        energy_au = sum(system._cpp.get_eigenenergies()[idx] for system, idx in zip(systems, ket_indices))

        self._cpp = self._cpp_type.create(*[basis._cpp for basis in basis_atoms], *ket_indices, energy_au)  # type: ignore [call-arg,arg-type]

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
    _cpp_type = _backend.KetPairReal  # type: ignore [assignment]
