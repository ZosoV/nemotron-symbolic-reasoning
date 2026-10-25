"""Add mask_clues to a CoT parquet for loss masking during training.

Adds two columns to the input parquet and writes a new file:
  mask_clues       — list of {mask_begin_clue, mask_end_clue} dicts consumed by
                     render_and_mask() in end-to-end-curriculum_learning.ipynb.
  cot_1_5          — cot column (rewritten; optionally tag-stripped).
  dataset_category — "examples_{n}" where n is parsed from the input filename.

Mask modes
----------
prefix        : mask user turn (prompt + boxed instruction) up to <think>\\n (1 clue).
prefix+dm1    : prefix + first digit_mapping_1 group (2 clues, deterministic).
prefix+midcot : prefix + one random factual_observations bullet + one random
                digit_mapping_N group (3 clues, seeded per row).

--strip-tags removes all structural XML delimiters from the CoT before rendering.
--strip-tags-keep-conclusion removes all structural XML delimiters except <conclusion>
    / </conclusion>, so the conclusion block is preserved during training.

Rows with no tagged sections (e.g. concatenation-only problems) that require
mid-CoT clues fall back silently to prefix-only masking.

Run:
    uv run python src/data_generation/add_mask_clues.py
    uv run python src/data_generation/add_mask_clues.py \\
        --input inputs/data/dataset_examples_5_k_2_balanced_cot.parquet \\
        --mask-mode prefix+dm1 --strip-tags
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ---------------------------------------------------------------------------
# Constants — MUST stay in sync with end-to-end-curriculum_learning.ipynb
# (search for BOXED_INSTRUCTION in that notebook to find the sync point)
# ---------------------------------------------------------------------------

BOXED_INSTRUCTION = (
    "Please put your final answer inside `\\boxed{}`. "
    "For example: `\\boxed{your answer}`"
)

CLUE_MIN = 80   # initial clue window in chars
CLUE_MAX = 400  # maximum before raising

# Structural tags that --strip-tags removes from the CoT.
# Each tag is expected to appear on its own line (possibly with leading spaces/tabs).
_STRUCT_TAG_RE = re.compile(
    r"[ \t]*</?("
    r"initial_reasoning"
    r"|available_operations"
    r"|operator_\d+_candidates"
    r"|factual_observations_operator_\d+"
    r"|digit_priority"
    r"|digit_mapping_\d+"
    r"|digit_mapping_analysis_\d+"
    r"|digit_mapping_analysis_follow_up_\d+"
    r"|conclusion"
    r")>[ \t]*\n?"
)

# Same as above but keeps <conclusion> / </conclusion> intact.
_STRUCT_TAG_KEEP_CONCLUSION_RE = re.compile(
    r"[ \t]*</?("
    r"initial_reasoning"
    r"|available_operations"
    r"|operator_\d+_candidates"
    r"|factual_observations_operator_\d+"
    r"|digit_priority"
    r"|digit_mapping_\d+"
    r"|digit_mapping_analysis_\d+"
    r"|digit_mapping_analysis_follow_up_\d+"
    r")>[ \t]*\n?"
)


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

def render_full_text(prompt: str, cot: str, answer: str) -> str:
    """Build the full Nemotron chat text exactly as the training consumer does.

    Mirrors apply_chat_template(enable_thinking=False) + _rearrange_think from
    end-to-end-curriculum_learning.ipynb cell 5.
    """
    user_content = f"{prompt}\n{BOXED_INSTRUCTION}"
    return (
        f"<|im_start|>system\n<|im_end|>\n"
        f"<|im_start|>user\n{user_content}<|im_end|>\n"
        f"<|im_start|>assistant\n<think>\n"
        f"{cot} \n</think>\n\\boxed{{{answer}}}\n<|im_end|>"
    )


def strip_struct_tags(text: str, keep_conclusion: bool = False) -> str:
    """Remove structural CoT delimiters, collapsing excessive blank lines.

    When keep_conclusion=True, <conclusion> / </conclusion> tags are preserved.
    """
    pattern = _STRUCT_TAG_KEEP_CONCLUSION_RE if keep_conclusion else _STRUCT_TAG_RE
    result = pattern.sub("\n", text)
    result = re.sub(r"\n{3,}", "\n\n", result)
    return result


# ---------------------------------------------------------------------------
# Clue building
# ---------------------------------------------------------------------------

def _unique_clue(full_text: str, anchor: int, *, direction: int) -> str:
    """Return shortest substring (starting from CLUE_MIN) that is unique in full_text.

    direction=+1: extend rightward from anchor  (begin clue).
    direction=-1: extend leftward  to anchor    (end clue).
    Raises ValueError if not achievable within CLUE_MAX chars.
    """
    for n in range(CLUE_MIN, CLUE_MAX + 1):
        s = full_text[anchor: anchor + n] if direction > 0 else full_text[max(0, anchor - n): anchor]
        if full_text.count(s) == 1:
            return s
    raise ValueError(
        f"Clue not unique within {CLUE_MAX} chars at anchor={anchor} "
        f"(direction={'right' if direction > 0 else 'left'})"
    )


def make_span_clue(full_text: str, abs_start: int, abs_end: int) -> dict[str, str]:
    """Return {mask_begin_clue, mask_end_clue} anchoring [abs_start, abs_end) in full_text.

    The consumer interprets these as:
        masked_region = [find(begin_clue), find(end_clue) + len(end_clue))
    """
    return {
        "mask_begin_clue": _unique_clue(full_text, abs_start, direction=+1),
        "mask_end_clue":   _unique_clue(full_text, abs_end,   direction=-1),
    }


def prefix_end_pos(full_text: str) -> int:
    """Return char offset just after '<|im_start|>assistant\\n<think>\\n'."""
    marker = "<|im_start|>assistant\n<think>\n"
    idx = full_text.index(marker)
    return idx + len(marker)


# ---------------------------------------------------------------------------
# Position mapping between original and stripped cot
# ---------------------------------------------------------------------------

def _cot_pos_in_final(cot: str, pos: int, do_strip: bool) -> int:
    """Map a char position in the original cot to its position in the (stripped) cot_final.

    When do_strip=False, the identity mapping is returned.
    When do_strip=True, we measure how many chars strip_struct_tags emits for cot[:pos].
    This relies on the fact that blank-line collapse is boundary-consistent
    (the prefix slice and the full slice agree on the collapsed result up to pos).
    """
    if not do_strip:
        return pos
    return len(strip_struct_tags(cot[:pos]))


# ---------------------------------------------------------------------------
# CoT span parsers (always operate on the *original* tagged cot)
# ---------------------------------------------------------------------------

_DM_OPEN_RE = re.compile(r"<digit_mapping_(\d+)>")


def _parse_dm_groups(cot: str) -> list[tuple[int, int, int]]:
    """Return list of (orig_start, orig_end, N) for each digit_mapping_N group.

    The span covers <digit_mapping_N> … </digit_mapping_analysis_follow_up_N>.
    """
    groups: list[tuple[int, int, int]] = []
    for m in _DM_OPEN_RE.finditer(cot):
        n = int(m.group(1))
        close_tag = f"</digit_mapping_analysis_follow_up_{n}>"
        c = cot.find(close_tag, m.start())
        if c == -1:
            continue
        groups.append((m.start(), c + len(close_tag), n))
    return groups


def _parse_factual_bullets(cot: str) -> list[tuple[int, int]]:
    """Return (orig_start, orig_end) for every bullet in factual_observations blocks.

    The span includes the leading '\\n' before '- ' and extends to end of the line.
    """
    bullets: list[tuple[int, int]] = []
    for block_m in re.finditer(
        r"<factual_observations_operator_\d+>(.*?)</factual_observations_operator_\d+>",
        cot, re.DOTALL,
    ):
        block_content = block_m.group(1)
        block_inner_start = block_m.start(1)
        for bm in re.finditer(r"\n(- [^\n]+)", block_content):
            # bm.start() is the \n; bm.end() is after the bullet text
            bullets.append((block_inner_start + bm.start(), block_inner_start + bm.end()))
    return bullets


# ---------------------------------------------------------------------------
# Per-row processing
# ---------------------------------------------------------------------------

def _process_row(
    prompt: str,
    cot: str,
    answer: str,
    mode: str,
    do_strip: bool,
    rng: "random.Random",
    keep_conclusion: bool = False,
) -> dict[str, Any]:
    """Return dict with keys 'cot_final' and 'mask_clues'."""
    import random as _random  # local import keeps top-level clean

    # 1. Build the final cot (tag-stripped or original)
    cot_final = strip_struct_tags(cot, keep_conclusion=keep_conclusion) if do_strip else cot

    # 2. Build the full rendered text (used only to locate clue substrings)
    full_text = render_full_text(prompt, cot_final, answer)

    # 3. Prefix clue: span = [0, prefix_end_pos(full_text))
    pend = prefix_end_pos(full_text)
    prefix_clue = make_span_clue(full_text, 0, pend)
    clues = [prefix_clue]

    # 4. Additional clues for mid-CoT modes
    if mode in ("prefix+dm1", "prefix+midcot"):
        # Parse from original tagged cot
        dm_groups = _parse_dm_groups(cot)

        if mode == "prefix+dm1":
            # Deterministic: always the N=1 group
            dm1 = next((g for g in dm_groups if g[2] == 1), None)
            if dm1 is not None:
                orig_s, orig_e, _ = dm1
                cot_s = _cot_pos_in_final(cot, orig_s, do_strip)
                cot_e = _cot_pos_in_final(cot, orig_e, do_strip)
                # Map cot-relative positions to full_text positions
                cot_offset = pend  # prefix_end_pos is exactly the start of cot in full_text
                clue = make_span_clue(full_text, cot_offset + cot_s, cot_offset + cot_e)
                clues.append(clue)
            # else: no dm1 group (concatenation row) — fall back to prefix-only silently

        else:  # prefix+midcot
            bullets = _parse_factual_bullets(cot)

            chosen_clues_added = 0

            if bullets:
                b_orig_s, b_orig_e = rng.choice(bullets)
                b_cot_s = _cot_pos_in_final(cot, b_orig_s, do_strip)
                b_cot_e = _cot_pos_in_final(cot, b_orig_e, do_strip)
                cot_offset = pend
                clue = make_span_clue(full_text, cot_offset + b_cot_s, cot_offset + b_cot_e)
                clues.append(clue)
                chosen_clues_added += 1

            if dm_groups:
                dm_orig_s, dm_orig_e, _ = rng.choice(dm_groups)
                dm_cot_s = _cot_pos_in_final(cot, dm_orig_s, do_strip)
                dm_cot_e = _cot_pos_in_final(cot, dm_orig_e, do_strip)
                cot_offset = pend
                clue = make_span_clue(full_text, cot_offset + dm_cot_s, cot_offset + dm_cot_e)
                clues.append(clue)
                chosen_clues_added += 1

    return {"cot_final": cot_final, "mask_clues": clues}


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

_EXPECTED_CLUES = {"prefix": 1, "prefix+dm1": (1, 2), "prefix+midcot": (1, 2, 3)}


def validate(df: pd.DataFrame, mode: str, cot_col: str) -> None:
    """Structural checks — raise AssertionError on failure."""
    expected = _EXPECTED_CLUES[mode]
    if isinstance(expected, int):
        expected = (expected,)

    bad_count = 0
    for i, row in df.iterrows():
        clues = row["mask_clues"]
        if not isinstance(clues, (list, tuple)) or len(clues) not in expected:
            bad_count += 1
            continue

        cot = str(row[cot_col])
        answer = str(row["answer"])
        prompt = str(row["prompt"])
        full_text = render_full_text(prompt, cot, answer)

        for cl in clues:
            b_pos = full_text.find(cl["mask_begin_clue"])
            e_pos = full_text.find(cl["mask_end_clue"])
            assert b_pos != -1, f"Row {i}: begin_clue not found: {cl['mask_begin_clue']!r}"
            assert e_pos != -1, f"Row {i}: end_clue not found: {cl['mask_end_clue']!r}"
            # Uniqueness
            assert full_text.count(cl["mask_begin_clue"]) == 1, (
                f"Row {i}: begin_clue not unique"
            )
            assert full_text.count(cl["mask_end_clue"]) == 1, (
                f"Row {i}: end_clue not unique"
            )

    assert bad_count == 0, f"{bad_count} rows have unexpected clue count for mode={mode!r}"
    print(f"  validate: {len(df)} rows OK (mode={mode!r})")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_n_from_path(path: Path) -> int | None:
    """Try to extract the shot-count from filenames like 'dataset_examples_5_...' → 5."""
    m = re.search(r"examples_(\d+)", path.stem)
    return int(m.group(1)) if m else None


def main(argv: list[str] | None = None) -> None:
    import random

    parser = argparse.ArgumentParser(
        description="Add mask_clues column to a CoT parquet for loss-masked training."
    )
    parser.add_argument(
        "--input",
        default=str(_ROOT / "inputs/data/dataset_examples_5_k_2_balanced_cot.parquet"),
        help="Source parquet with prompt, cot_col, answer columns.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output parquet path (default: <input_stem>_masked.parquet).",
    )
    parser.add_argument(
        "--mask-mode",
        choices=["prefix", "prefix+dm1", "prefix+midcot"],
        default="prefix",
        help="Masking strategy (default: prefix).",
    )
    parser.add_argument(
        "--strip-tags",
        action="store_true",
        default=False,
        help="Remove all structural CoT XML tags before rendering.",
    )
    parser.add_argument(
        "--strip-tags-keep-conclusion",
        action="store_true",
        default=False,
        help="Remove structural CoT XML tags except <conclusion> / </conclusion>.",
    )
    parser.add_argument(
        "--cot-col",
        default="cot_1_5",
        help="Name of the CoT column (default: cot_1_5).",
    )
    parser.add_argument("--seed", type=int, default=0, help="RNG seed (default: 0).")
    args = parser.parse_args(argv)

    if args.strip_tags and args.strip_tags_keep_conclusion:
        raise SystemExit("--strip-tags and --strip-tags-keep-conclusion are mutually exclusive.")

    keep_conclusion = args.strip_tags_keep_conclusion
    do_strip = args.strip_tags or keep_conclusion

    input_path = Path(args.input)
    if args.output is None:
        stem = input_path.stem
        if keep_conclusion:
            suffix = "_stripped_keep_conclusion_masked"
        elif args.strip_tags:
            suffix = "_stripped_masked"
        else:
            suffix = "_masked"
        out_path = input_path.with_name(stem + suffix + ".parquet")
    else:
        out_path = Path(args.output)

    print(f"Loading {input_path} …")
    df = pd.read_parquet(input_path)
    print(f"  Loaded {len(df)} rows  columns: {list(df.columns)}")

    if args.cot_col not in df.columns:
        raise SystemExit(f"Column {args.cot_col!r} not found. Available: {list(df.columns)}")

    # Add dataset_category (required by the curriculum training consumer)
    if "dataset_category" in df.columns:
        print(f"  dataset_category: using existing column ({df['dataset_category'].nunique()} unique value(s))")
    else:
        n_shot = _parse_n_from_path(input_path)
        if n_shot is not None:
            df["dataset_category"] = f"examples_{n_shot}"
            print(f"  dataset_category: set to 'examples_{n_shot}' (parsed from filename)")
        else:
            print("  WARNING: dataset_category not in input and could not parse shot count from filename; column not added")

    # Drop rows with missing/empty cot
    before = len(df)
    df = df[df[args.cot_col].apply(lambda v: isinstance(v, str) and bool(v.strip()))].reset_index(drop=True)
    dropped = before - len(df)
    print(f"  Dropped {dropped} rows with missing/empty {args.cot_col!r} ({len(df)} remaining)")

    # Process rows
    rng = random.Random(args.seed)
    mask_clues_col: list[list[dict]] = []
    cot_col_out: list[str] = []

    for idx, row in df.iterrows():
        cot_raw = row[args.cot_col]

        # Seed per-row rng for reproducibility regardless of row order
        row_rng = random.Random(args.seed ^ hash(str(row.get("problem_id", idx))))

        try:
            result = _process_row(
                prompt=str(row["prompt"]),
                cot=cot_raw,
                answer=str(row["answer"]),
                mode=args.mask_mode,
                do_strip=do_strip,
                rng=row_rng,
                keep_conclusion=keep_conclusion,
            )
        except Exception as exc:
            print(f"  WARNING row {idx}: {exc} — using prefix-only fallback")
            try:
                cot_final = strip_struct_tags(cot_raw, keep_conclusion=keep_conclusion) if do_strip else cot_raw
                full_text = render_full_text(str(row["prompt"]), cot_final, str(row["answer"]))
                pend = prefix_end_pos(full_text)
                clues = [make_span_clue(full_text, 0, pend)]
                result = {"cot_final": cot_final, "mask_clues": clues}
            except Exception as exc2:
                print(f"  WARNING row {idx}: fallback also failed ({exc2}) — empty clues")
                result = {"cot_final": cot_raw, "mask_clues": []}

        mask_clues_col.append(result["mask_clues"])
        cot_col_out.append(result["cot_final"])

    df[args.cot_col] = cot_col_out
    df["mask_clues"] = mask_clues_col

    print("Running validation …")
    validate(df, args.mask_mode, args.cot_col)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"Saved → {out_path}  ({len(df)} rows, {len(df.columns)} columns)")

    # Summary
    clue_counts = df["mask_clues"].apply(len).value_counts().sort_index()
    print("  mask_clues length distribution:")
    for n, cnt in clue_counts.items():
        print(f"    {n} clues: {cnt} rows")


if __name__ == "__main__":
    main()
