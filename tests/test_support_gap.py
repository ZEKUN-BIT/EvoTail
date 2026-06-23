from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evotail.analysis.support_gap import (
    bootstrap_stratum_means,
    fit_support_regression,
    run_support_gap_analysis,
    spearman_correlation,
    summarize_assays,
)


class SupportGapTests(unittest.TestCase):
    def setUp(self) -> None:
        rows = []
        for assay_id, protein_id, support, stratum, relationship in [
            ("tail_assay", "tail", 0.1, 0, -1.0),
            ("head_assay", "head", 0.9, 3, 1.0),
        ]:
            for index in range(24):
                phenotype = float(index)
                rows.append(
                    {
                        "DMS_id": assay_id,
                        "protein_id": protein_id,
                        "mutant": f"A{index + 1}C",
                        "DMS_score": phenotype,
                        "evo_score": relationship * phenotype,
                        "evolutionary_support_score": support,
                        "support_stratum": stratum,
                        "sequence_length": 100,
                    }
                )
        self.variants = pd.DataFrame(rows)

    def test_spearman_and_assay_summary(self) -> None:
        self.assertAlmostEqual(spearman_correlation(pd.Series([1, 2, 3]), pd.Series([3, 2, 1])), -1.0)
        summary = summarize_assays(self.variants, min_variants=20)
        self.assertEqual(len(summary), 2)
        self.assertAlmostEqual(summary.loc[summary.DMS_id == "head_assay", "spearman"].item(), 1.0)
        self.assertAlmostEqual(summary.loc[summary.DMS_id == "tail_assay", "spearman"].item(), -1.0)

    def test_bootstrap_and_regression_return_structured_output(self) -> None:
        summary = summarize_assays(self.variants, min_variants=20)
        bootstrap = bootstrap_stratum_means(summary, samples=100, seed=1)
        self.assertEqual(bootstrap["n_assays"].sum(), 2)
        regression = fit_support_regression(summary)
        self.assertIn("n_assays", regression)

    def test_end_to_end_analysis_writes_outputs(self) -> None:
        support = self.variants[["protein_id", "evolutionary_support_score", "support_stratum", "sequence_length"]].drop_duplicates()
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            scores_path = root / "scores.csv"
            support_path = root / "support.csv"
            self.variants[["DMS_id", "protein_id", "mutant", "DMS_score", "evo_score"]].to_csv(scores_path, index=False)
            support.to_csv(support_path, index=False)
            stats = run_support_gap_analysis(scores_path, support_path, root / "out", min_variants=20, bootstrap_samples=100)
            self.assertEqual(stats["n_assays"], 2)
            self.assertTrue((root / "out" / "per_assay_spearman.csv").is_file())
            persisted = json.loads((root / "out" / "support_gap_stats.json").read_text())
            self.assertIn("support_gap", persisted)


if __name__ == "__main__":
    unittest.main()

