"""Assemble the full tagged CoT for one feasible puzzle row.

Pulls together every section module and emits the template's tag structure:
  <initial_reasoning> <available_operations>
  <operator_N_candidates> + <factual_observations_operator_N>
  <digit_priority>
  <digit_mapping_N> / <digit_mapping_analysis_N> / <digit_mapping_analysis_follow_up_N>
  <conclusion>

Run (self-test, prints a full CoT for the template puzzle):
    uv run python src/cot_generator/render.py
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.cot_generator.analysis import AttemptAnalysis, analyze_attempt, solution_program
from src.cot_generator.config import CotGeneratorConfig
from src.cot_generator.decode import (
    LITTLE_ENDIAN, concat_pattern, decode_operand, detect_pure_concat, predict_answer,
)
from src.cot_generator.digit_priority import compute_digit_priority
from src.cot_generator.equations import (
    Equation, content_symbols, operator_groups, parse_prompt,
)
from src.cot_generator.factual_observations import OperatorAnalysis, analyze_operators
from src.cot_generator.families import (
    FAMILY_BY_NAME, FAMILIES, apply_family, int_to_base_digits, resolve_registry_op,
)
from src.cot_generator.perturbation import build_chain
from src.cot_generator.redundancy_observations import apply_redundancies
from src.cot_generator.tells import make_tell


@dataclass
class CotResult:
    text: str
    answer: str                 # predicted answer string
    mode: str
    program: dict[str, tuple[str, int]]
    num_perturbations: int = 0  # failed attempts before the solution (chain length - 1)


def _tag(name: str, body: str) -> str:
    return f"<{name}>\n{body}\n</{name}>"


def _fmt_off(off: int) -> str:
    return "0" if off == 0 else f"{off:+d}"


# ---------------------------------------------------------------------------
# Static-ish sections
# ---------------------------------------------------------------------------

def _initial_reasoning(examples, q, operand_syms, operator_syms) -> str:
    eq_lines = "\n".join(f"- eq[{e.index}]: {e.raw}" for e in examples)
    return (
        f"The problem requires deciphering the answer for the left-hand side (lhs) "
        f"{q.l0}{q.l1}{q.op}{q.r0}{q.r1} given some example equations that act as "
        f"constraints. We label the example equations eq[1], eq[2], … and refer to them "
        f"by these tags below. In this case, we have:\n\n{eq_lines}\n\n"
        f"Operand symbols: {', '.join(operand_syms)}\n\n"
        f"Operator symbols: {', '.join(operator_syms)}\n\n"
        "Each equation says its lhs equals its rhs once the operation is applied. "
        "We assume the puzzle is consistent — at least one combination satisfies "
        "every example equation: a **digit mapping** sending each symbol to a unique "
        "digit (no two symbols share a digit), and an **operator mapping** sending "
        "each operator symbol to one arithmetic operation (several operations may fit, "
        "but only one is ultimately used).\n\n"
        "Magnitudes are decoded with one of two strategies, abbreviated std and le below:\n"
        "- `standard` (std): AB = A x base + B\n"
        "- `little_endian` (le): AB = B x base + A (a digit swap after standard decode)\n\n"
        "We always use base 10. Results are decoded most-significant digit first for "
        "`standard` (std); for `little_endian` (le) the result digit list is reversed."
    )


def _available_operations() -> str:
    lines = ["Possible candidates for any operator symbol:"]
    for f in FAMILIES:
        if f.has_offset:
            offs = ", ".join(_fmt_off(o) for o in f.offsets)
            lines.append(f"- {f.name}(a, b) = {f.formula}, offset in {{{offs}}}")
        else:
            lines.append(f"- {f.name}(a, b) = {f.formula}")
    lines.append(
        "\n**Offset rule:** start at offset 0; if a result is close to the target by a small "
        "constant in the allowed set, fix that offset and carry it forward for every later "
        "check of the same op."
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Operator candidate blocks
# ---------------------------------------------------------------------------

def _render_operator_block(n: int, a: OperatorAnalysis, q) -> str:
    role = " (operator symbol of the question)" if a.is_question_op else ""
    obs_lines = "\n".join(f"- {o.text}" for o in a.observations) or "- (no structural eliminations)"
    inner_obs = _tag(f"factual_observations_operator_{n}", obs_lines)
    if not a.constrained:
        body = (
            f"Analysis for operator symbol `{a.op}`{role}: no example equation contains it.\n\n"
            f"{inner_obs}\n\n"
            f"Remaining candidates for `{a.op}`:\n- all operations (unconstrained by the examples)"
        )
        return _tag(f"operator_{n}_candidates", body)

    eq_lines = "\n".join(f"- eq[{e.index}]: {e.raw}" for e in a.eqs)
    remaining = []
    for fam in a.candidate_families:
        offs = a.survivors[fam]
        if FAMILY_BY_NAME[fam].has_offset and offs != FAMILY_BY_NAME[fam].offsets:
            remaining.append(f"{fam} (offsets {{{', '.join(_fmt_off(o) for o in offs)}}})")
        else:
            remaining.append(fam)
    body = (
        f"Analysis for operator symbol `{a.op}`{role}, the following equations contain it:\n\n"
        f"{eq_lines}\n\n{inner_obs}\n\n"
        f"Remaining candidates for `{a.op}`:\n- {', '.join(remaining)}"
    )
    return _tag(f"operator_{n}_candidates", body)


# ---------------------------------------------------------------------------
# Digit-mapping attempt blocks
# ---------------------------------------------------------------------------

def _render_mapping_first(step, priority_order, base) -> str:
    pool = list(range(base))
    lines = ["Assigning a mapping by priority:\n"]
    for s in priority_order:
        d = step.mapping[s]
        lines.append(f"- from {{{', '.join(map(str, pool))}}} picking: {s} = {d}")
        pool.remove(d)
    if step.unused:
        lines.append(f"\nDigit candidates not yet used: {{{', '.join(map(str, step.unused))}}}")
    return "\n".join(lines)


def _render_mapping_next(step, priority_order) -> str:
    fix = step.fixes_into  # set by caller: the fix that PRODUCED this mapping
    if fix is None:
        change = ""
    elif fix.kind == "swap":
        a, b = fix.symbols
        change = f" (swapping {a} <-> {b})"
    else:
        s = fix.symbols[0]
        change = f" (reassigning {s} to {step.mapping[s]})"
    lines = [f"Following the previous observations, let's try this mapping{change}:\n"]
    for s in priority_order:
        lines.append(f"- {s} = {step.mapping[s]}")
    if step.unused:
        lines.append(f"\nDigit candidates not yet used: {{{', '.join(map(str, step.unused))}}}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Conclusion
# ---------------------------------------------------------------------------

def _answer_working(q, solution, fam, off, mode, base, answer) -> str:
    """Show the decode -> apply -> encode steps that produce the boxed answer."""
    lhs = f"{q.l0}{q.l1}{q.op}{q.r0}{q.r1}"
    head = f"Using the feasible mappings to answer the question {lhs} (`{mode}`):"
    if fam in ("concat_fwd", "concat_rev"):
        pat = "".join(concat_pattern(fam, q))
        return (f"{head}\n"
                f"- {fam} copies the operand symbols in written order: {lhs} -> {pat} = {answer}")

    L = decode_operand(q.l0, q.l1, solution, base, mode)
    R = decode_operand(q.r0, q.r1, solution, base, mode)
    dl0, dl1 = solution[q.l0], solution[q.l1]
    dr0, dr1 = solution[q.r0], solution[q.r1]
    value = apply_family(fam, off, L, R, base)
    off_txt = f" (offset {_fmt_off(off)})" if FAMILY_BY_NAME[fam].has_offset else ""
    digits = int_to_base_digits(abs(value), base)
    if len(digits) < q.res_len:
        digits = [0] * (q.res_len - len(digits)) + digits
    nat = ",".join(str(d) for d in digits)
    if mode == LITTLE_ENDIAN:
        enc_txt = f"digits {nat} reversed to {','.join(str(d) for d in reversed(digits))}"
    else:
        enc_txt = f"digits {nat}"
    return (f"{head}\n"
            f"- decode: {q.l0}{q.l1} -> {dl0},{dl1} -> {L}; "
            f"{q.r0}{q.r1} -> {dr0},{dr1} -> {R}\n"
            f"- apply: {fam}({L}, {R}){off_txt} = {value}\n"
            f"- encode: |{value}| -> {enc_txt} -> {answer}")


def _render_conclusion(q, solution, program, mode, base, priority_order, answer) -> str:
    digit_lines = "\n".join(f"- {s} = {solution[s]}" for s in priority_order)
    op_lines = []
    for op, (fam, off) in program.items():
        op_lines.append(f"- `{op}` -> {fam}(a, b) = {FAMILY_BY_NAME[fam].formula}, offset {_fmt_off(off)}")
    fam, off = program[q.op]
    return (
        "The following mappings are feasible:\n\n"
        f"digit mapping:\n{digit_lines}\n\n"
        f"operation mapping:\n" + "\n".join(op_lines) + "\n\n"
        f"strategy: `{mode}` encoding with base {base}\n\n"
        f"{_answer_working(q, solution, fam, off, mode, base, answer)}\n\n"
        f"The final answer is \\box{{{answer}}}\n\n"
        "Notice: this is one valid answer. Other combinations may also satisfy all constraints."
    )


# ---------------------------------------------------------------------------
# Pure-concat CoT (concise variant — no digit mapping)
# ---------------------------------------------------------------------------

def render_concat_cot(
    examples: list[Equation], q: Equation, fam: str,
) -> CotResult:
    """Render a short, tagged CoT for a pure-concat row (no digit mapping needed).

    Three sections:
      <initial_reasoning>  — puzzle layout, concat signal, structural insight
      <concat_evidence>    — per-example verification of the concat direction
      <conclusion>         — apply the pattern and emit \\box{answer}
    """
    # --- initial_reasoning ------------------------------------------------
    eq_lines = "\n".join(f"- eq[{e.index}]: {e.raw}" for e in examples)
    operand_syms = content_symbols(examples + [q])
    operator_syms = sorted({eq.op for eq in examples} | {q.op})
    ir = (
        f"The problem requires deciphering the answer for the left-hand side (lhs) "
        f"{q.l0}{q.l1}{q.op}{q.r0}{q.r1} given some example equations that act as "
        f"constraints. We label the example equations eq[1], eq[2], … and refer to them "
        f"by these tags below. In this case, we have:\n\n{eq_lines}\n\n"
        f"Operand symbols: {', '.join(operand_syms)}\n\n"
        f"Operator symbols: {', '.join(operator_syms)}\n\n"
        f"Each equation says its lhs equals its rhs once the operation is applied.\n\n"
        f"Observation: every example equation that uses operator `{q.op}` reproduces "
        f"its result by placing the operand symbols side-by-side in a fixed order — "
        f"that is *pure concatenation*, which needs no hidden digit mapping or encoding. "
        f"We verify this holds across all relevant examples, then apply the same pattern "
        f"to the question lhs."
    )

    # --- concat_evidence --------------------------------------------------
    relevant = [e for e in examples if e.op == q.op]
    if fam == "concat_fwd":
        direction_label = "forward (l0 l1 r0 r1)"
        pattern_desc = f"l0 l1 r0 r1"
    else:
        direction_label = "reverse (r0 r1 l0 l1)"
        pattern_desc = f"r0 r1 l0 l1"

    ev_lines = []
    for e in relevant:
        pat = "".join(concat_pattern(fam, e))
        ev_lines.append(
            f"- eq[{e.index}]: {e.raw}  →  rhs = {''.join(e.rhs)} = {pat} ✓ ({fam})"
        )
    ev_body = (
        f"Checking operator `{q.op}` ({direction_label} pattern: {pattern_desc}):\n\n"
        + "\n".join(ev_lines)
        + f"\n\nAll {len(relevant)} example equation(s) with `{q.op}` are satisfied by "
        f"{fam}; the direction is consistent — `{q.op}` = {fam}."
    )

    # --- conclusion -------------------------------------------------------
    pat_syms = concat_pattern(fam, q)
    answer = "".join(pat_syms)
    pat_str = " ".join(pat_syms)
    conc = (
        f"Apply {fam} to the question lhs {q.l0}{q.l1}{q.op}{q.r0}{q.r1}:\n"
        f"  operands → {pat_str} → {answer}\n\n"
        f"The final answer is \\box{{{answer}}}"
    )

    text = "\n\n".join([
        _tag("initial_reasoning", ir),
        _tag("concat_evidence", ev_body),
        _tag("conclusion", conc),
    ])
    return CotResult(text=text, answer=answer, mode="concat", program={q.op: (fam, 0)})


# ---------------------------------------------------------------------------
# Top-level
# ---------------------------------------------------------------------------

def render_cot(row: dict, cfg: CotGeneratorConfig, rounds: int | None = None) -> CotResult:
    """Render the full CoT for one feasible solver-results row.

    `rounds` overrides `cfg.perturbation_rounds` when given (used by the
    token-cap retry loop in generate.py to shrink the CoT one perturbation at a
    time). The actual number of perturbations realised — capped by the puzzle
    structure, which may be below the request — is reported on the returned
    `CotResult.num_perturbations`.
    """
    solution = json.loads(row["digits_mapping"])
    om = json.loads(row["ops_mapping"])
    base = int(round(float(row["solver_radix"])))
    problem_id = str(row.get("id", "")).split("_")[0]

    examples, q = parse_prompt(row["prompt"], row["answer"])
    operand_syms = content_symbols(examples + [q])
    operator_syms = sorted({eq.op for eq in examples} | {q.op})

    # Anchor to the solver's actual program so the conclusion reproduces the
    # labelled answer (puzzles are underdetermined).
    mode = "little_endian" if "little" in str(row["solver_strategy"]).lower() else "standard"
    solver_prog = {op: resolve_registry_op(name) for op, name in om.items()}

    analyses = analyze_operators(examples, q.op, base)
    # Protect the question operator's solver family from the redundancy drop.
    apply_redundancies(analyses, protect={q.op: solver_prog[q.op][0]})

    program = solution_program(analyses, solution, base, mode, solver_prog)

    # Digit-priority order (drives the rendering of digit-mapping blocks).
    # Unconstrained operators contribute nothing, so exclude them from scoring.
    constrained = [a for a in analyses if a.constrained]
    dp = compute_digit_priority(
        constrained, examples, operand_syms, cfg.digit_priority_mode,
        cfg.random_seed, problem_id,
    )
    priority_order = dp.order

    # Backward perturbation chain (mappings fail under `program` until the solution).
    chain = build_chain(
        solution, operand_syms, examples, program, base,
        cfg.perturbation_rounds if rounds is None else rounds,
        cfg.random_seed, problem_id, mode,
    )
    # Tag each step with the fix that produced it (for the change phrase).
    for i, st in enumerate(chain.steps):
        st.fixes_into = None if i == 0 else chain.steps[i - 1].fixes[0]

    parts: list[str] = []
    parts.append(_tag("initial_reasoning",
                     _initial_reasoning(examples, q, operand_syms, operator_syms)))
    parts.append(
        "First, let's identify the possible operators and discard infeasible ones based "
        "solely on the form and structure of the example equations:\n\n"
        + _tag("available_operations", _available_operations()))
    parts.append(
        "Let's analyze the operator symbols one by one and make factual observations based "
        "on the form of the equations:")
    for n, a in enumerate(analyses, start=1):
        parts.append(_render_operator_block(n, a, q))

    parts.append("Now, let's analyze the possible digit mapping (a unique digit 0-9 per symbol).")
    parts.append(_tag("digit_priority", dp.text))

    prev_map = None
    prev_attempt: AttemptAnalysis | None = None
    for i, st in enumerate(chain.steps):
        n = st.index
        if i == 0:
            parts.append(_tag(f"digit_mapping_{n}", _render_mapping_first(st, priority_order, base)))
        else:
            parts.append(_tag(f"digit_mapping_{n}", _render_mapping_next(st, priority_order)))
        attempt = analyze_attempt(n, st.mapping, prev_map, analyses, prev_attempt, base, program)
        parts.append(_tag(f"digit_mapping_analysis_{n}", attempt.text))
        follow = make_tell(st, examples, solution, program, base, mode, cfg.tell_mode)
        parts.append(_tag(f"digit_mapping_analysis_follow_up_{n}", follow))
        prev_map = st.mapping
        prev_attempt = attempt

    answer = predict_answer(q, *program[q.op], solution, base, mode)
    parts.append(_tag("conclusion",
                     _render_conclusion(q, solution, program, mode, base, priority_order, answer)))

    return CotResult(text="\n\n".join(parts), answer=answer, mode=mode, program=program,
                     num_perturbations=len(chain.steps) - 1)


def _selftest() -> None:
    import pandas as pd
    df = pd.read_parquet("inputs/data/result_examples_3_rand_k_6_max2.parquet")
    row = df[df["id"] == "00457d26_q0"].iloc[0].to_dict()
    cfg = CotGeneratorConfig(digit_priority_mode="most_constrained", tell_mode="algebraic",
                            perturbation_rounds=4, random_seed=0)
    res = render_cot(row, cfg)
    assert res.answer == row["solver_answer"], (res.answer, row["solver_answer"])
    for tag in ("initial_reasoning", "available_operations", "operator_1_candidates",
                "digit_priority", "digit_mapping_1", "digit_mapping_analysis_follow_up_1",
                "conclusion"):
        assert f"<{tag}>" in res.text, f"missing <{tag}>"
    print("render self-test PASS  answer =", res.answer)
    print("=" * 80)
    print(res.text)


if __name__ == "__main__":
    _selftest()
