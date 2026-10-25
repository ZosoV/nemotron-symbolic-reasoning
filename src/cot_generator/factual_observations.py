"""Script #1 - per-operator factual observations.

Thin orchestration layer over feasibility.py: group the example equations by
operator glyph, run the structural family filter on each group, and package the
result for rendering the <operator_N_candidates> / <factual_observations_operator_N>
blocks. The question's operator is reported first (operator_1), matching the
template.

The heavy lifting (sign / concat / digit-count / self-application checks, and the
folded-in solver heuristics #2 and #4) lives in feasibility.py.

Run (self-test):
    uv run python src/cot_generator/factual_observations.py
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.cot_generator.equations import Equation, operator_groups
from src.cot_generator.feasibility import Observation, feasible_for_operator


@dataclass
class OperatorAnalysis:
    """Per-operator candidate analysis for one CoT."""
    op: str
    eqs: list[Equation]
    is_question_op: bool
    survivors: dict[str, tuple[int, ...]]      # family -> surviving offsets
    observations: list[Observation] = field(default_factory=list)

    @property
    def candidate_families(self) -> list[str]:
        return list(self.survivors.keys())

    @property
    def constrained(self) -> bool:
        return len(self.eqs) > 0


def analyze_operators(
    examples: list[Equation], question_op: str, base: int,
) -> list[OperatorAnalysis]:
    """Run Script #1 for every operator, question operator first."""
    groups = operator_groups(examples)
    ordered_ops = [question_op] + [op for op in groups if op != question_op]

    out: list[OperatorAnalysis] = []
    for op in ordered_ops:
        eqs = groups.get(op, [])
        # Keep the question operator even if no example constrains it (it is
        # settled by the question alone); skip other operators with no equations.
        if not eqs and op != question_op:
            continue
        feas = feasible_for_operator(op, eqs, base)
        out.append(OperatorAnalysis(
            op=op,
            eqs=eqs,
            is_question_op=(op == question_op),
            survivors=feas.survivors,
            observations=list(feas.observations),
        ))
    return out


def _selftest() -> None:
    from src.cot_generator.equations import parse_prompt

    prompt = (
        "In Alice's Wonderland, a secret set of transformation rules is applied "
        "to equations. Below are a few examples:\n"
        "[[-!' = @&\n`!*\\& = '@'{\n\\'-!` = \\\\\n\n"
        "Now, determine the result for: `!*[{"
    )
    examples, q = parse_prompt(prompt, "'\"[`")
    analyses = analyze_operators(examples, q.op, base=10)
    assert analyses[0].op == "*" and analyses[0].is_question_op
    # 4-digit unsigned rhs whose symbols don't match a concat pattern: concat
    # dropped, neg_absdiff dropped, arithmetic kept.
    assert "neg_absdiff" not in analyses[0].survivors
    assert "concat_fwd" not in analyses[0].survivors and "concat_rev" not in analyses[0].survivors
    assert {"mul", "lcm"} <= set(analyses[0].survivors)
    minus = next(a for a in analyses if a.op == "-")
    # neg_absdiff + concat gone; sub/rsub still here (redundancy removes them later)
    assert "neg_absdiff" not in minus.survivors
    assert "sub_signed" in minus.survivors and "rsub_signed" in minus.survivors
    print("factual_observations self-test PASS")


if __name__ == "__main__":
    _selftest()
