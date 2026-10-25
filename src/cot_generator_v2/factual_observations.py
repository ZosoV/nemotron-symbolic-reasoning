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

from src.cot_generator_v2.equations import Equation, operator_groups
from src.cot_generator_v2.feasibility import Observation, feasible_for_operator


@dataclass
class OperatorAnalysis:
    """Per-operator candidate analysis for one CoT."""
    op: str
    eqs: list[Equation]
    is_question_op: bool
    survivors: dict[str, tuple[int, ...]]      # family -> surviving offsets (mode-agnostic)
    observations: list[Observation] = field(default_factory=list)
    # Per-mode view from OperatorFeasibility.survivors_by_mode.
    # mode -> {family -> surviving offsets}; a family absent from a mode was
    # structurally ruled out for that mode only.
    survivors_by_mode: dict[str, dict[str, tuple[int, ...]]] = field(
        default_factory=dict
    )

    @property
    def candidate_families(self) -> list[str]:
        return list(self.survivors.keys())

    @property
    def constrained(self) -> bool:
        return len(self.eqs) > 0


def analyze_operators(
    examples: list[Equation],
    question_op: str,
    base: int,
    protect: dict[str, str] | None = None,
) -> list[OperatorAnalysis]:
    """Run Script #1 for every operator, question operator first.

    `protect` maps operator glyph -> family name that must not be dropped by
    the sign heuristic (used to keep a gold op's family when it is sub_signed /
    rsub_signed but the example rhs values happen to be all unsigned).
    """
    protect = protect or {}
    groups = operator_groups(examples)
    ordered_ops = [question_op] + [op for op in groups if op != question_op]

    out: list[OperatorAnalysis] = []
    for op in ordered_ops:
        eqs = groups.get(op, [])
        # Keep the question operator even if no example constrains it (it is
        # settled by the question alone); skip other operators with no equations.
        if not eqs and op != question_op:
            continue
        op_protect = frozenset({protect[op]} if op in protect else [])
        feas = feasible_for_operator(op, eqs, base, protect=op_protect)
        # Default survivors_by_mode to identical views if not populated
        # (unconstrained operator path in feasible_for_operator).
        sbm = feas.survivors_by_mode or {
            mode: dict(feas.survivors) for mode in ["standard", "little_endian"]
        }
        out.append(OperatorAnalysis(
            op=op,
            eqs=eqs,
            is_question_op=(op == question_op),
            survivors=feas.survivors,
            observations=list(feas.observations),
            survivors_by_mode=sbm,
        ))
    return out


def _selftest() -> None:
    from src.cot_generator_v2.equations import parse_prompt

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
    # All three negative-capable families dropped by the sign check (no protect used).
    assert "neg_absdiff" not in minus.survivors
    assert "sub_signed" not in minus.survivors and "rsub_signed" not in minus.survivors
    print("factual_observations self-test PASS")


if __name__ == "__main__":
    _selftest()
