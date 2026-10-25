"""Dedup the non-cryptarithm replay corpus by problem_id (keep first).

Input:  inputs/data/v2/train_corpus_no_cryptarithm.parquet
Output: inputs/data/v2/train_corpus_no_cryptarithm_dedup.parquet
"""

import pathlib
import pandas as pd

IN_PATH  = pathlib.Path("inputs/data/v2/train_corpus_no_cryptarithm.parquet")
OUT_PATH = pathlib.Path("inputs/data/v2/train_corpus_no_cryptarithm_dedup.parquet")

df = pd.read_parquet(IN_PATH)
print(f"Input:  {len(df)} rows, {df['problem_id'].nunique()} unique problem_ids")

df_dedup = df.drop_duplicates(subset=["problem_id"], keep="first").reset_index(drop=True)
print(f"Output: {len(df_dedup)} rows, {df_dedup['problem_id'].nunique()} unique problem_ids")
print(f"Removed {len(df) - len(df_dedup)} duplicate rows")

df_dedup.to_parquet(OUT_PATH, index=False)
print(f"Saved → {OUT_PATH}")
