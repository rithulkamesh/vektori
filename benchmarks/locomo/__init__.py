"""Utilities for preparing LoCoMo benchmark datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Any

__all__ = ["cook_locomo_dataset"]


def cook_locomo_dataset(
	input_path: str | Path,
	output_path: str | Path,
	history_policy: str = "all_sessions",
	include_image_fields: bool = False,
) -> dict[str, int]:
	from benchmarks.locomo.cook_locomo import cook_locomo_dataset as _cook_locomo_dataset

	return _cook_locomo_dataset(
		input_path=input_path,
		output_path=output_path,
		history_policy=history_policy,
		include_image_fields=include_image_fields,
	)
