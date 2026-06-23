from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evotail.data.cluster_sequences import build_mmseqs_easy_cluster_command, cluster_sizes_from_tsv
from evotail.data.support_score import SupportScoreConfig, SupportScoreError, compute_support_table


class SupportScoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.metadata = pd.DataFrame(
            {
                "protein_id": ["tail", "mid", "head", "missing"],
                "sequence": ["ACDE", "ACDEF", "ACDEFG", "ACDEFGH"],
                "nearest_identity": [0.05, 0.45, 0.95, np.nan],
                "cluster_size_50": [1, 10, 1000, np.nan],
                "msa_neff": [0, 5, 500, np.nan],
                "taxon_count": [1, 2, 20, np.nan],
            }
        )

    def test_support_score_is_finite_and_tail_gets_more_weight(self) -> None:
        output = compute_support_table(self.metadata, SupportScoreConfig(strata=2))
        self.assertTrue(np.isfinite(output["evolutionary_support_score"]).all())
        self.assertTrue(np.isfinite(output["tail_sampling_weight"]).all())
        self.assertGreater(
            output.loc[output.protein_id == "tail", "tail_sampling_weight"].item(),
            output.loc[output.protein_id == "head", "tail_sampling_weight"].item(),
        )
        self.assertEqual(output.loc[output.protein_id == "missing", "support_missing_component_count"].item(), 4)

    def test_observed_zero_msa_is_valid_but_negative_is_not(self) -> None:
        output = compute_support_table(self.metadata)
        self.assertEqual(output.loc[output.protein_id == "tail", "support_msa_neff"].item(), 0)
        invalid = self.metadata.copy()
        invalid.loc[0, "msa_neff"] = -1
        with self.assertRaises(SupportScoreError):
            compute_support_table(invalid)

    def test_invalid_identity_fails_loudly(self) -> None:
        invalid = self.metadata.copy()
        invalid.loc[0, "nearest_identity"] = 1.1
        with self.assertRaises(SupportScoreError):
            compute_support_table(invalid)

    def test_mmseqs_command_and_cluster_sizes(self) -> None:
        command = build_mmseqs_easy_cluster_command("in.faa", "out", "tmp", 0.5)
        self.assertEqual(command[:2], ["mmseqs", "easy-cluster"])
        self.assertIn("--min-seq-id", command)
        path = Path(self._testMethodName + ".tsv")
        self.addCleanup(path.unlink, missing_ok=True)
        path.write_text("rep1\tp1\nrep1\tp2\nrep2\tp3\n", encoding="utf-8")
        sizes = cluster_sizes_from_tsv(path)
        self.assertEqual(sizes.loc[sizes.protein_id == "p1", "cluster_size"].item(), 2)
        self.assertEqual(sizes.loc[sizes.protein_id == "p3", "cluster_size"].item(), 1)


if __name__ == "__main__":
    unittest.main()
