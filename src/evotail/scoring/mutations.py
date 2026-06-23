"""Strict parsing and validation of ProteinGym substitution strings."""

from __future__ import annotations

import re
from dataclasses import dataclass


MUTATION_PATTERN = re.compile(r"^([ACDEFGHIKLMNPQRSTVWY])(\d+)([ACDEFGHIKLMNPQRSTVWY])$")


@dataclass(frozen=True)
class Substitution:
    wild_type: str
    position: int
    mutant: str


def parse_mutant(value: object) -> list[Substitution]:
    """Parse `A1P:D2N` style ProteinGym substitutions; WT is an empty mutation set."""

    text = str(value).strip().upper()
    if text in {"", "WT", "NAN"}:
        return []
    mutations: list[Substitution] = []
    for token in text.split(":"):
        match = MUTATION_PATTERN.fullmatch(token)
        if match is None:
            raise ValueError(f"Invalid substitution token: {token!r}")
        wild_type, position, mutant = match.groups()
        mutations.append(Substitution(wild_type, int(position), mutant))
    if len({mutation.position for mutation in mutations}) != len(mutations):
        raise ValueError(f"Mutation repeats a position: {value!r}")
    return mutations


def apply_mutations(target_sequence: str, mutations: list[Substitution]) -> str:
    """Apply validated substitutions and verify every supplied wild-type residue."""

    sequence = list(target_sequence.strip().upper())
    for mutation in mutations:
        index = mutation.position - 1
        if index < 0 or index >= len(sequence):
            raise ValueError(f"Mutation position {mutation.position} exceeds target length")
        if sequence[index] != mutation.wild_type:
            raise ValueError(
                f"Mutation expects {mutation.wild_type}{mutation.position}, found {sequence[index]}"
            )
        sequence[index] = mutation.mutant
    return "".join(sequence)

