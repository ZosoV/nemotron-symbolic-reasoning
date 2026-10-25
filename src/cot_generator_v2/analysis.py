"""Render the <digit_mapping_analysis_N> block for one attempt.

For each operator (question operator first) and each candidate family that
survived the structural observations, test the family under both decode modes
across the operator's equations. A family is valid in a mode iff a single offset
(carried forward) satisfies every equation in that mode. The block then narrows
the encoding mode and concludes whether the attempt yields a feasible program.

Token-saving conventions, per the spec:
  * reduced one-line-per-candidate format;
  * when an operator's symbols are unchanged from the previous attempt, the
    analysis is not recomputed - it states the carried-over conclusion.

Run (self-test):
    uv run python src/cot_generator/analysis.py
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.cot_generator_v2.decode import (
    LITTLE_ENDIAN, MODES, STANDARD, decode_operand, evaluate_equation, family_matches,
)
from src.cot_generator_v2.equations import Equation
from src.cot_generator_v2.factual_observations import OperatorAnalysis
from src.cot_generator_v2.families import FAMILY_BY_NAME, apply_family


def _fmt_off(off: int) -> str:
    return "0" if off == 0 else f"{off:+d}"


def _mode_tag(mode: str) -> str:
    return "std" if mode == STANDARD else "le"


def _fmtval(v) -> str:
    return str(v) if v is not None else "guard-fail"


def _decode_line(eq: Equation, digit_map, base: int, mode: str) -> str:
    """One-line operand/target decode for an equation under `mode`."""
    L = decode_operand(eq.l0, eq.l1, digit_map, base, mode)
    R = decode_operand(eq.r0, eq.r1, digit_map, base, mode)
    dl0, dl1 = digit_map[eq.l0], digit_map[eq.l1]
    dr0, dr1 = digit_map[eq.r0], digit_map[eq.r1]
    ev = evaluate_equation(eq, digit_map, base, mode)
    tgt = ",".join(str(d) for d in ev.target_digits)
    sign = ", signed" if eq.has_sign else ""
    return (f"{eq.l0}{eq.l1} -> {dl0},{dl1} -> {L}; "
            f"{eq.r0}{eq.r1} -> {dr0},{dr1} -> {R}; "
            f"result {''.join(eq.rhs)} -> {tgt} (target {ev.target}{sign})")


def _candidate_mode_line(
    family: str, offsets, eq: Equation, digit_map, base: int, mode: str,
) -> tuple[str, list[int]]:
    """Render one (family, mode) check against `eq`; return (line, surviving_offsets)."""
    if family in ("concat_fwd", "concat_rev"):
        mr = family_matches(family, 0, eq, digit_map, base, mode)
        if mr.ok:
            return ("reproduces the operand symbols in the rhs -> correct", list(offsets))
        return (f"{mr.reason} -> incorrect", [])

    ev = evaluate_equation(eq, digit_map, base, mode)
    results = [(o, family_matches(family, o, eq, digit_map, base, mode)) for o in offsets]
    surviving = [o for o, mr in results if mr.ok]
    has_off = FAMILY_BY_NAME[family].has_offset
    tgt = ",".join(str(d) for d in ev.target_digits)

    if surviving:
        o = surviving[0]
        mr = dict(results)[o]
        off_txt = f" (offset {_fmt_off(o)})" if has_off else ""
        return (f"{family}({ev.L}, {ev.R}) = {_fmtval(mr.value)} -> {tgt} -> "
                f"correct{off_txt}", surviving)

    if has_off and len(offsets) > 1:
        base_val = apply_family(family, 0, ev.L, ev.R, base)
        sweep = ", ".join(_fmtval(apply_family(family, o, ev.L, ev.R, base)) for o in offsets)
        offs_txt = ", ".join(_fmt_off(o) for o in offsets)
        return (f"{family}({ev.L}, {ev.R}) = {_fmtval(base_val)}; offsets {{{offs_txt}}} -> "
                f"{{{sweep}}}, none equal target {ev.target} -> incorrect", [])

    o, mr = results[0]
    return (f"{family}({ev.L}, {ev.R}) = {_fmtval(mr.value)} -> "
            f"{mr.reason or 'value != target'} -> incorrect", [])


def _offset_hint(family: str, live: dict, live_modes: list[str]) -> str:
    if not FAMILY_BY_NAME[family].has_offset:
        return ""
    offs = sorted({o for m in live_modes if family in live[m] for o in live[m][family]})
    if len(offs) == 1:
        return f" (offset {_fmt_off(offs[0])})"
    return f" (offset in {{{', '.join(_fmt_off(o) for o in offs)}}})"


def _survivors_desc(live: dict) -> str:
    parts = []
    for mode in MODES:
        tag = _mode_tag(mode)
        if mode in live and live[mode]:
            fam_descs = []
            for fam, offs in live[mode].items():
                if FAMILY_BY_NAME[fam].has_offset:
                    od = (_fmt_off(offs[0]) if len(offs) == 1
                          else "{" + ", ".join(_fmt_off(o) for o in offs) + "}")
                    fam_descs.append(f"{fam}(offset {od})")
                else:
                    fam_descs.append(fam)
            parts.append(f"{tag}: " + ", ".join(fam_descs))
        else:
            parts.append(f"{tag}: eliminated")
    return "; ".join(parts)


def _count_pairs(live: dict) -> int:
    """Count total surviving (family, offset) pairs across all modes."""
    total = 0
    for mode_live in live.values():
        for offs in mode_live.values():
            total += len(offs)
    return total


def _detailed_op_block(
    a: OperatorAnalysis, digit_map, base: int,
    program: dict[str, tuple[str, int]] | None,
) -> tuple[str, dict[str, tuple[str, int]]]:
    """Template-faithful narrowing analysis for one operator.

    Fix 5 early-stop rules (implemented):
    2a: all (family, offset) pairs for this operator eliminated → infeasible, note and stop.
    2b: exactly one (family, offset) pair remains across all modes → determined, stop
        checking further equations for this operator.
    3:  one encoding mode eliminated → note it, stop testing that mode.
    """
    op = a.op
    # Seed per-mode candidates from the structural per-mode view when available
    # (families structurally dead in one mode are excluded from that mode's live
    # set so they are never evaluated there). Fall back to the mode-agnostic
    # survivors for operators without a per-mode view.
    sbm = a.survivors_by_mode if a.survivors_by_mode else {}
    live: dict[str, dict[str, list[int]]] = {
        mode: (
            {fam: list(offs) for fam, offs in sbm[mode].items()}
            if mode in sbm
            else {fam: list(a.survivors[fam]) for fam in a.candidate_families}
        )
        for mode in MODES
    }
    body: list[str] = [f"Analysis for `{op}`:"]

    for eq in a.eqs:
        live_modes = [m for m in MODES if live.get(m)]
        if not live_modes:
            break  # 2a already fired
        body.append("")
        body.append(f"eq[{eq.index}]: {eq.raw}")
        for mode in live_modes:
            body.append(f"- {_mode_tag(mode)}: {_decode_line(eq, digit_map, base, mode)}")

        live_fams = [f for f in a.candidate_families if any(f in live[m] for m in live_modes)]
        for fam in live_fams:
            body.append(f"Checking {fam}{_offset_hint(fam, live, live_modes)}:")
            for mode in live_modes:
                if fam not in live[mode]:
                    continue
                line, surviving = _candidate_mode_line(
                    fam, live[mode][fam], eq, digit_map, base, mode)
                body.append(f"- {_mode_tag(mode)}: {line}")
                if surviving:
                    live[mode][fam] = surviving
                else:
                    del live[mode][fam]

        # Fix 5 Rule 3: drop mode if all its families are gone; note if newly gone
        prev_mode_count = len(live_modes)
        for mode in list(live):
            if not live[mode]:
                del live[mode]
        if len(live) < prev_mode_count and live:
            surviving_mode = next(iter(live))
            body.append(f"eq[{eq.index}] eliminates {_mode_tag('little_endian' if surviving_mode == STANDARD else STANDARD)} "
                        f"for `{op}`. Encoding fixed to {_mode_tag(surviving_mode)}.")

        body.append(f"Survivors after eq[{eq.index}]: {_survivors_desc(live)}")

        # Fix 5 Rule 2a: no pairs left → infeasible
        if not live:
            body.append(f"No candidate survives for `{op}` on eq[{eq.index}]. This mapping is infeasible.")
            break

        # Fix 5 Rule 2b: exactly one (family, offset) pair remains → determined
        if _count_pairs(live) == 1:
            mode = next(iter(live))
            fam = next(iter(live[mode]))
            off = live[mode][fam][0]
            body.append(f"After eq[{eq.index}]: only {fam} (offset {_fmt_off(off)}, "
                        f"{_mode_tag(mode)}) survives for `{op}`. Operator determined. "
                        "Carrying forward.")
            break  # stop checking further equations for this operator

    # Collect surviving modes' (family, offset), preferring the solver's program.
    valid_modes: dict[str, tuple[str, int]] = {}
    for mode in MODES:
        if not live.get(mode):
            continue
        chosen: tuple[str, int] | None = None
        if program and op in program:
            pfam, poff = program[op]
            if pfam in live[mode] and poff in live[mode][pfam]:
                chosen = (pfam, poff)
        if chosen is None:
            fam = next(iter(live[mode]))
            chosen = (fam, live[mode][fam][0])
        valid_modes[mode] = chosen

    if valid_modes:
        modes_txt = " and ".join(_mode_tag(m) for m in MODES if m in valid_modes)
        fam, off = next(iter(valid_modes.values()))
        body.append(f"=> `{op}` VALID in {modes_txt} (e.g. {fam}, offset {_fmt_off(off)}).")
    else:
        body.append(f"=> no candidate validates `{op}` under this mapping (attempt fails).")
    return "\n".join(body), valid_modes


@dataclass
class OpAttempt:
    op: str
    unchanged: bool
    valid_modes: dict[str, tuple[str, int]]   # mode -> (family, offset) that validates
    text: str


@dataclass
class AttemptAnalysis:
    index: int
    is_solution: bool
    consistent_modes: list[str]               # modes where every operator validates
    text: str
    op_attempts: list[OpAttempt] = field(default_factory=list)
    is_infeasible: bool = False               # Fix 5: True when an operator lost all candidates


def _op_symbols(eqs: list[Equation]) -> set[str]:
    out: set[str] = set()
    for eq in eqs:
        out.update(eq.symbols)
    return out


def analyze_attempt(
    index: int,
    digit_map: dict[str, int],
    prev_map: dict[str, int] | None,
    analyses: list[OperatorAnalysis],
    prev_attempt: "AttemptAnalysis | None",
    base: int,
    program: dict[str, tuple[str, int]] | None = None,
) -> AttemptAnalysis:
    """Analyse one digit mapping across all operators and both modes."""
    op_attempts: list[OpAttempt] = []
    lines: list[str] = []

    for ai, a in enumerate(analyses):
        # Unconstrained operator (no example equation): nothing to check.
        if not a.eqs:
            txt = (f"Analysis for `{a.op}`: no example equation constrains it, so it places no "
                   "restriction on this mapping.")
            lines.append(txt)
            op_attempts.append(OpAttempt(a.op, True, {m: ("", 0) for m in MODES}, txt))
            continue

        op_syms = _op_symbols(a.eqs)
        unchanged = (
            prev_map is not None
            and prev_attempt is not None
            and ai < len(prev_attempt.op_attempts)  # guard: early-exit may truncate
            and all(digit_map.get(s) == prev_map.get(s) for s in op_syms)
        )
        if unchanged:
            prev_op = prev_attempt.op_attempts[ai]
            modes_txt = " and ".join(prev_op.valid_modes) if prev_op.valid_modes else "neither mode"
            if prev_op.valid_modes:
                fam, off = next(iter(prev_op.valid_modes.values()))
                concl = f"{fam} (offset {_fmt_off(off)}) remains VALID in {modes_txt}"
            else:
                concl = "still no VALID candidate"
            syms_txt = ", ".join(sorted(op_syms))
            lines.append(
                f"Analysis for `{a.op}`: equations use only {syms_txt} - unchanged from "
                f"the previous attempt -> {concl}."
            )
            op_attempts.append(OpAttempt(a.op, True, dict(prev_op.valid_modes), lines[-1]))
            continue

        # Recompute from scratch — detailed, narrowing analysis (Option A).
        text, valid_modes = _detailed_op_block(a, digit_map, base, program)
        lines.append(text)
        op_attempts.append(OpAttempt(a.op, False, valid_modes, text))

        # Fix 5 Rule 2a: operator has no surviving candidates → infeasible, stop now.
        if not valid_modes:
            return AttemptAnalysis(
                index=index, is_solution=False, consistent_modes=[],
                text="\n\n".join(lines), op_attempts=op_attempts,
                is_infeasible=True,
            )

    consistent = [
        m for m in MODES if all(m in oa.valid_modes for oa in op_attempts)
    ]
    is_solution = bool(consistent)
    if is_solution:
        lines.append(
            f"All operators validate under {' and '.join(consistent)} encoding -> this mapping "
            "is a feasible solution."
        )
    else:
        # Name an operator that failed for the follow-up to act on.
        failed = [oa.op for oa in op_attempts if not oa.valid_modes]
        if failed:
            lines.append(
                f"No consistent encoding works: operator(s) {', '.join(failed)} have no valid "
                "candidate, so at least one of their symbols holds a wrong digit."
            )
    return AttemptAnalysis(
        index=index, is_solution=is_solution, consistent_modes=consistent,
        text="\n\n".join(lines), op_attempts=op_attempts,
    )


def solution_program(
    analyses: list[OperatorAnalysis], solution: dict[str, int], base: int,
    mode: str, prefer: dict[str, tuple[str, int]],
) -> dict[str, tuple[str, int]]:
    """Per-operator (family, offset) that validates the solution under `mode`.

    Anchored to the solver's program: for each operator the solver's own
    (family, offset) is tried first (`prefer`), so the conclusion reproduces the
    row's labelled answer; otherwise any surviving family that validates is used
    (puzzles are underdetermined and several feasible ops may exist). Raises
    ValueError if some operator has no validating survivor.
    """
    program: dict[str, tuple[str, int]] = {}
    for a in analyses:
        chosen: tuple[str, int] | None = None
        pf = prefer.get(a.op)
        # Try the solver's exact (family, offset) first.
        if pf is not None and pf[0] in a.survivors and pf[1] in a.survivors[pf[0]]:
            if all(family_matches(pf[0], pf[1], eq, solution, base, mode).ok for eq in a.eqs):
                chosen = pf
        if chosen is None:
            for fam in a.candidate_families:
                for off in a.survivors[fam]:
                    if all(family_matches(fam, off, eq, solution, base, mode).ok for eq in a.eqs):
                        chosen = (fam, off)
                        break
                if chosen is not None:
                    break
        if chosen is None:
            raise ValueError(f"operator `{a.op}` has no validating survivor under {mode}")
        program[a.op] = chosen
    return program


def _selftest() -> None:
    import json
    import pandas as pd
    from src.cot_generator_v2.equations import parse_prompt
    from src.cot_generator_v2.factual_observations import analyze_operators
    from src.cot_generator_v2.redundancy_observations import apply_redundancies
    from src.cot_generator_v2.families import resolve_registry_op

    df = pd.read_parquet("inputs/data/v2/only_cryptharithm_solved_k_1.parquet")
    r = df.iloc[0]
    sol = json.loads(r["digits_mapping"])
    om = json.loads(r["ops_mapping"])
    base = int(round(float(r["solver_radix"])))
    mode = str(r["solver_strategy"]).strip().lower()
    examples, q = parse_prompt(r["prompt"], r["answer"])
    analyses = analyze_operators(examples, q.op, base)
    solver_prog = {op: resolve_registry_op(n) for op, n in om.items()}
    apply_redundancies(analyses, protect={q.op: solver_prog[q.op][0]})
    program = solution_program(analyses, sol, base, mode, solver_prog)

    # Solution mapping must analyse as a feasible solution.
    res = analyze_attempt(5, sol, None, analyses, None, base, program)
    assert res.is_solution, res.text
    print("analysis self-test PASS  program=", program)
    print(res.text)


if __name__ == "__main__":
    _selftest()
