"""Assemble and write the v2 parquets.

Produces two files under output_dir/v2/:
  dataset_v2_base.parquet         — base random-drop pools
  dataset_v2_augmentation.parquet — gap-fill augmentation (Strategy A + B)

Both carry a `dataset_category` column (e.g. "examples_3").  Downstream tools
(cot_generator/generate.py, add_mask_clues.py) consume these files unchanged.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.encoding import verify_program


def emit_parquets(
    base_samples: list[dict],
    aug_samples: list[dict],
    output_dir: str | Path,
) -> tuple[Path, Path]:
    """Validate and write both parquets. Returns (base_path, aug_path)."""
    out = Path(output_dir) / "v2"
    out.mkdir(parents=True, exist_ok=True)

    base_path = out / "dataset_v2_base.parquet"
    aug_path = out / "dataset_v2_augmentation.parquet"

    print(f"Validating {len(base_samples)} base samples...")
    base_ok = _validate_samples(base_samples)
    print(f"  passed: {base_ok}/{len(base_samples)}")

    print(f"Validating {len(aug_samples)} augmentation samples...")
    aug_ok = _validate_samples(aug_samples)
    print(f"  passed: {aug_ok}/{len(aug_samples)}")

    base_df = pd.DataFrame(base_samples)
    aug_df = pd.DataFrame(aug_samples)

    # Bucket summary
    for label, df in [("base", base_df), ("augmentation", aug_df)]:
        if "dataset_category" in df.columns:
            print(f"\n{label} bucket counts:")
            print(df["dataset_category"].value_counts().sort_index().to_string())

    base_df.to_parquet(base_path, index=False)
    aug_df.to_parquet(aug_path, index=False)
    print(f"\nWrote {base_path}")
    print(f"Wrote {aug_path}")

    return base_path, aug_path


def _validate_samples(samples: list[dict]) -> int:
    """Run verify_program on arithmetic rows. Returns count of rows that passed."""
    passed = 0
    for row in samples:
        strategy = str(row.get("solver_strategy") or "")
        if strategy == "pure_concat":
            passed += 1
            continue
        dm_raw = row.get("digits_mapping")
        om_raw = row.get("ops_mapping")
        eq_raw = row.get("equations")
        if dm_raw is None or om_raw is None:
            passed += 1  # treat as pure_concat or skip
            continue
        try:
            dm = json.loads(dm_raw) if isinstance(dm_raw, str) else dm_raw
            om = json.loads(om_raw) if isinstance(om_raw, str) else om_raw
            eqs = json.loads(eq_raw) if isinstance(eq_raw, str) else eq_raw
        except (json.JSONDecodeError, TypeError):
            continue
        radix = int(row.get("solver_radix") or 10)
        mode = "little_endian" if strategy == "little_endian" else "standard"
        if verify_program(eqs, dm, om, mode, radix):
            passed += 1
    return passed
