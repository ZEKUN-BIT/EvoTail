from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evotail.scoring.mutations import apply_mutations, parse_mutant


class MutationTests(unittest.TestCase):
    def test_parse_and_apply_multiple_mutations(self) -> None:
        mutations = parse_mutant("A1C:D3Y")
        self.assertEqual(len(mutations), 2)
        self.assertEqual(apply_mutations("ACDE", mutations), "CCYE")

    def test_wild_type_and_invalid_inputs(self) -> None:
        self.assertEqual(parse_mutant("WT"), [])
        with self.assertRaises(ValueError):
            parse_mutant("A1C:A1D")
        with self.assertRaises(ValueError):
            parse_mutant("not-a-mutation")


if __name__ == "__main__":
    unittest.main()
