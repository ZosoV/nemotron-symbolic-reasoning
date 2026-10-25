"""CotGeneratorConfig dataclass and YAML loader.

Reads the `cot_generator:` block of a shared config.yaml. Self-contained — it
does not depend on src/config.py (SolverConfig). All fields are optional;
defaults are shown below.

YAML keys (under `cot_generator:`):
  input_parquet: str        — solver-results .parquet to read (feasible rows)
  output_parquet: str       — where to write the parquet with the new CoT column
  cot_column: str           — name of the generated column (default 'cot_1_5')
  digit_priority_mode: str  — 'most_constrained' | 'occurrences' | 'both'
                              'both' picks one per sample with a seeded RNG
  perturbation_rounds: int  — number of failed attempts before the solution
                              (chain length = rounds + 1); capped by what the
                              puzzle structure allows
  tell_mode: str            — 'algebraic' (operand-gap + result-position tells)
                              | 'simple' (name the failing position and fix)
  random_seed: int          — base seed for reproducible random choices
  limit: int|null           — process at most N feasible rows (null = all)
  max_cot_tokens: int|null  — cap on cot_approx_tokens. When a rendered CoT
                              exceeds it, the row is re-rendered with one fewer
                              perturbation and re-checked, repeating until it
                              fits or no perturbations remain (null = no cap)

Run (self-test):
    uv run python src/cot_generator/config.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PRIORITY_MODES = ("most_constrained", "occurrences", "both")
TELL_MODES = ("algebraic", "simple")


@dataclass
class CotGeneratorConfig:
    input_parquet: str = "inputs/data/result_examples_3_rand_k_6_max2.parquet"
    output_parquet: str = "inputs/data/result_examples_3_cot_1_5.parquet"
    cot_column: str = "cot_1_5"
    digit_priority_mode: str = "both"
    perturbation_rounds: int = 4
    tell_mode: str = "algebraic"
    random_seed: int = 0
    limit: int | None = None
    max_cot_tokens: int | None = None

    def __post_init__(self) -> None:
        if self.digit_priority_mode not in PRIORITY_MODES:
            raise ValueError(
                f"digit_priority_mode must be one of {PRIORITY_MODES}, "
                f"got {self.digit_priority_mode!r}"
            )
        if self.tell_mode not in TELL_MODES:
            raise ValueError(f"tell_mode must be one of {TELL_MODES}, got {self.tell_mode!r}")


def load_cot_config(path: str | Path | None = None) -> CotGeneratorConfig:
    """Load CotGeneratorConfig from the `cot_generator:` block of a YAML file."""
    if path is None:
        return CotGeneratorConfig()

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    import yaml  # lazy import so the module loads without pyyaml

    with open(path) as f:
        data = (yaml.safe_load(f) or {}).get("cot_generator", {}) or {}

    raw_limit = data.get("limit", None)
    raw_cap = data.get("max_cot_tokens", None)
    defaults = CotGeneratorConfig()
    return CotGeneratorConfig(
        input_parquet=str(data.get("input_parquet", defaults.input_parquet)),
        output_parquet=str(data.get("output_parquet", defaults.output_parquet)),
        cot_column=str(data.get("cot_column", defaults.cot_column)),
        digit_priority_mode=str(data.get("digit_priority_mode", defaults.digit_priority_mode)).lower(),
        perturbation_rounds=int(data.get("perturbation_rounds", defaults.perturbation_rounds)),
        tell_mode=str(data.get("tell_mode", defaults.tell_mode)).lower(),
        random_seed=int(data.get("random_seed", defaults.random_seed)),
        limit=int(raw_limit) if raw_limit is not None else None,
        max_cot_tokens=int(raw_cap) if raw_cap is not None else None,
    )


def _selftest() -> None:
    c = CotGeneratorConfig()
    assert c.cot_column == "cot_1_5" and c.digit_priority_mode == "both"
    try:
        CotGeneratorConfig(tell_mode="bogus")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for bad tell_mode")
    print("config self-test PASS")


if __name__ == "__main__":
    _selftest()
