"""Helpers that attach Quokka-specific metadata and utilities to pyAMReX MultiFabs."""

from __future__ import annotations

from typing import Sequence, Tuple, Union

import numpy as np

Direction = Union[int, str]


def _direction_index(direction: Direction, ndim: int) -> int:
	"""Map a user-friendly direction to an integer axis."""
	if isinstance(direction, int):
		if direction < 0 or direction >= ndim:
			raise ValueError(f"Direction index {direction} outside valid range [0, {ndim})")
		return direction

	if isinstance(direction, str):
		order = ("x", "y", "z")
		try:
			idx = order.index(direction.lower())
		except ValueError as exc:
			raise ValueError(f"Unsupported direction string '{direction}'") from exc
		if idx >= ndim:
			raise ValueError(f"Direction '{direction}' not available for {ndim}D data")
		return idx

	raise TypeError(f"Expected direction as int or str, got {type(direction)}")


def _extract_metadata(mf) -> Tuple[Sequence[float], Sequence[float]]:
	dx = getattr(mf, "_quokka_dx", None)
	prob_lo = getattr(mf, "_quokka_prob_lo", None)

	if dx is None or prob_lo is None:
		raise AttributeError(
		    "MultiFab is missing Quokka geometry metadata. "
		    "Please obtain it through pyquokka simulation accessors."
		)

	return tuple(dx), tuple(prob_lo)


def mesh(self, direction: Direction, include_ghosts: bool = False) -> np.ndarray:
	"""Return the physical mesh coordinates along the requested direction."""
	dx, prob_lo = _extract_metadata(self)
	axis = _direction_index(direction, len(dx))
	imesh = self.imesh(axis, include_ghosts)
	return prob_lo[axis] + imesh * dx[axis]


def cell_centers(self, include_ghosts: bool = False) -> Tuple[np.ndarray, ...]:
	"""Return per-dimension arrays containing the physical cell centers."""
	dx, prob_lo = _extract_metadata(self)
	centers = []
	for axis in range(len(dx)):
		imesh = self.imesh(axis, include_ghosts)
		centers.append(prob_lo[axis] + imesh * dx[axis])
	return tuple(centers)


def register_multifab_extension(amr) -> None:
	"""Attach Quokka-aware helpers to the MultiFab type provided by pyAMReX."""
	multifab_type = amr.MultiFab
	if getattr(multifab_type, "_pyquokka_multifab_registered", False):
		return

	setattr(multifab_type, "_pyquokka_multifab_registered", True)
	# Instance attributes are populated from C++, but default these so attribute
	# access does not fail before a simulation attaches metadata.
	multifab_type._quokka_dx = None
	multifab_type._quokka_prob_lo = None

	multifab_type.mesh = mesh
	multifab_type.cell_centers = cell_centers
