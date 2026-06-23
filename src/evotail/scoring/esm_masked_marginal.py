"""Per-assay ESM masked-marginal scoring for ProteinGym substitutions."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

import pandas as pd

from evotail.scoring.mutations import Substitution, apply_mutations, parse_mutant


@dataclass(frozen=True)
class VariantRecord:
    row_index: int
    sequence: str
    substitution: Substitution


class EsmMaskedMarginalScorer:
    """Score variants as a sum of masked-token log likelihood ratios.

    The context is the full mutant sequence with one mutated position masked at a time.
    This is an explicit, reproducible baseline; it should be compared with ProteinGym's
    official scoring conventions before claiming leaderboard equivalence.
    """

    def __init__(
        self,
        model_name: str = "facebook/esm2_t30_150M_UR50D",
        device: str = "cuda",
        cache_dir: str | None = None,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForMaskedLM, AutoTokenizer
        except ImportError as error:  # pragma: no cover - GPU dependency
            raise RuntimeError("Install torch and transformers to score ProteinGym assays") from error
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available; pass --device cpu only for smoke tests")
        self.torch = torch
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name, cache_dir=cache_dir).to(self.device)
        self.model.eval()
        if self.tokenizer.mask_token_id is None:
            raise RuntimeError("Tokenizer does not define a mask token")

    @staticmethod
    def _records(variants: pd.DataFrame, target_sequence: str) -> list[VariantRecord]:
        records: list[VariantRecord] = []
        for row_index, row in variants.iterrows():
            mutations = parse_mutant(row["mutant"])
            sequence = row.get("mutated_sequence")
            if pd.isna(sequence) or not str(sequence).strip():
                sequence = apply_mutations(target_sequence, mutations)
            sequence = str(sequence).strip().upper()
            if len(sequence) != len(target_sequence):
                raise ValueError(f"Variant row {row_index} changes sequence length; substitutions only are supported")
            for substitution in mutations:
                if target_sequence[substitution.position - 1] != substitution.wild_type:
                    raise ValueError(f"Variant row {row_index} does not match target sequence")
                records.append(VariantRecord(row_index, sequence, substitution))
        return records

    def score_variants(
        self, variants: pd.DataFrame, target_sequence: str, batch_size: int = 32
    ) -> pd.Series:
        if "mutant" not in variants:
            raise ValueError("variants must contain a mutant column")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        target_sequence = target_sequence.strip().upper()
        records = self._records(variants, target_sequence)
        scores: dict[int, float] = defaultdict(float)
        for start in range(0, len(records), batch_size):
            batch = records[start : start + batch_size]
            sequences = [record.sequence for record in batch]
            encoded = self.tokenizer(sequences, padding=True, return_tensors="pt")
            input_ids = encoded["input_ids"].clone()
            for row, record in enumerate(batch):
                token_position = record.substitution.position  # ESM position 0 is BOS.
                input_ids[row, token_position] = self.tokenizer.mask_token_id
            encoded["input_ids"] = input_ids.to(self.device)
            encoded["attention_mask"] = encoded["attention_mask"].to(self.device)
            with self.torch.inference_mode():
                logits = self.model(**encoded).logits
                log_probabilities = self.torch.log_softmax(logits, dim=-1)
            for row, record in enumerate(batch):
                token_position = record.substitution.position
                mutant_id = self.tokenizer.convert_tokens_to_ids(record.substitution.mutant)
                wild_type_id = self.tokenizer.convert_tokens_to_ids(record.substitution.wild_type)
                delta = log_probabilities[row, token_position, mutant_id] - log_probabilities[
                    row, token_position, wild_type_id
                ]
                scores[record.row_index] += float(delta.cpu())
        return pd.Series([scores[index] for index in variants.index], index=variants.index, dtype=float)

