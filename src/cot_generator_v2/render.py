"""Assemble the full tagged CoT for one feasible puzzle row.

Unified CoT structure — both concat and arithmetic puzzles share:
  <initial_reasoning>   — puzzle layout (short; no std/le preamble)
  <concat_check>        — always tries concat first
                          Outcome A (pure concat): done after <conclusion>
                          Outcome B (arithmetic):  transitions into the arithmetic path

Arithmetic path continues:
  <available_operations>
  <operator_N_candidates> + <factual_observations_operator_N>
  <digit_mapping_N> / <digit_mapping_analysis_N> / <digit_mapping_analysis_follow_up_N>
  <conclusion>

Run (self-test, prints a full CoT for the template puzzle):
    uv run python src/cot_generator_v2/render.py
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.cot_generator_v2.analysis import AttemptAnalysis, analyze_attempt, solution_program
from src.cot_generator_v2.config import CotGeneratorConfig
from src.cot_generator_v2.decode import (
    LITTLE_ENDIAN, MODES, STANDARD, concat_pattern, decode_operand,
    detect_pure_concat, predict_answer, render_concat_check_block,
)
from src.cot_generator_v2.digit_constraint import (
    picking_sequence,
    reduce_allowed_from_followup,
    render_first_mapping_block_occurrences,
)
from src.cot_generator_v2.digit_priority import compute_digit_priority
from src.cot_generator_v2.equations import (
    Equation, content_symbols, operator_groups, parse_prompt,
)
from src.cot_generator_v2.factual_observations import OperatorAnalysis, analyze_operators
from src.cot_generator_v2.families import (
    FAMILY_BY_NAME, FAMILIES, apply_family, int_to_base_digits, resolve_registry_op,
)
from src.cot_generator_v2.perturbation import build_chain
from src.cot_generator_v2.redundancy_observations import apply_redundancies
from src.cot_generator_v2.tells import make_tell


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
# Shared initial reasoning (short — std/le details deferred to arithmetic path)
# ---------------------------------------------------------------------------

def _render_initial_reasoning(examples, q, operand_syms, operator_syms) -> str:
    """Common puzzle layout. No std/le encoding details — added after concat check."""
    eq_lines = "\n".join(f"- eq[{e.index}]: {e.raw}" for e in examples)

    operand_display = ""
    if operand_syms['unconstrained']:
        operand_display = (f"{', '.join(operand_syms['constrained'])}"
                        f"\n\t- unconstrained wrt example equations: {', '.join(operand_syms['unconstrained'])}"
                        )
    else:
        operand_display = f"{', '.join(operand_syms['constrained'])}"

    return (
        f"The problem requires inferring the transformation rules to decipher the answer "
        f"for the left-hand side (lhs) {q.l0}{q.l1}{q.op}{q.r0}{q.r1} given some example "
        f"equations, labeled as:\n\n{eq_lines}\n\n"
        f"Operand symbols: {operand_display}\n\n"
        f"Operator symbols: {', '.join(operator_syms)}\n\n"
        f"Left-hand side to decipher: {q.l0}{q.l1}{q.op}{q.r0}{q.r1}\n\n"
        "Each equation says its lhs (composed by l0 l1 op r0 r1) equals its rhs once the operation is applied."
    )


def _render_encoding_explanation() -> str:
    """Std / little-endian explanation — shown only after concat is ruled out."""
    return (
        "The puzzle requires a **digit mapping** sending each symbol to a unique digit "
        "(no two symbols share a digit), and an **operator mapping** sending each operator "
        "symbol to one arithmetic operation.\n\n"
        "Magnitudes are decoded with one of two strategies, abbreviated std and le below:\n"
        "- `standard` (std): AB = A × base + B\n"
        "- `little_endian` (le): AB = B × base + A (a digit swap after standard decode)\n\n"
        "We always use base 10. Results are decoded most-significant digit first for "
        "`standard` (std); for `little_endian` (le) the result digit list is reversed."
    )


# ---------------------------------------------------------------------------
# Available operations — Fix 7/6: correct offsets, result constraints, offset rule
# ---------------------------------------------------------------------------

def _available_operations() -> str:
    """List every candidate operation with correct offsets, result constraints, and
    the Fix-6 offset rule: (family, offset) pairs are eliminated per-equation;
    one pair surviving = locked in; zero pairs = infeasible."""
    lines = ["Possible candidates for any operator symbol:"]
    for f in FAMILIES:
        if f.has_offset:
            offs = ", ".join(_fmt_off(o) for o in f.offsets)
            note = ""
            if f.name in ("sub_signed", "rsub_signed"):
                note = "  [result can be negative]"
            elif f.name == "neg_absdiff":
                note = "  [result always ≤ 0]"
            elif f.name == "absdiff":
                note = "  [result ≥ 0; guard: |a−b|+offset ≥ 0]"
            lines.append(f"- {f.name}(a, b) = {f.formula}, offset in {{{offs}}}{note}")
        else:
            note = ""
            if f.name in ("fdiv", "mod"):
                note = "  [guard: b ≠ 0]"
            elif f.name == "rmod":
                note = "  [guard: a ≠ 0]"
            elif f.name == "lcm":
                note = "  [= a×b // gcd(a,b) if (a and b) else 0]"
            lines.append(f"- {f.name}(a, b) = {f.formula}{note}")
    lines.append(
        "\n**Offset rule:** each `(operation, offset)` pair is treated as a separate "
        "candidate. A pair survives an equation only if `operation(L, R) + offset = target`. "
        "Pairs that fail any equation are eliminated. Once a single pair remains it is "
        "locked in and carried forward. If no pair survives, the family is eliminated."
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Operator candidate blocks
# ---------------------------------------------------------------------------

def _candidate_mode_label(fam: str, a: OperatorAnalysis) -> str:
    """Return the mode qualifier for a surviving family: 'both', 'standard',
    or 'little_endian', derived from a.survivors_by_mode."""
    sbm = a.survivors_by_mode
    in_std = fam in sbm.get(STANDARD, {})
    in_le  = fam in sbm.get(LITTLE_ENDIAN, {})
    if in_std and in_le:
        return "both"
    return STANDARD if in_std else LITTLE_ENDIAN


def _render_operator_block(n: int, a: OperatorAnalysis, q) -> str:
    obs_lines = "\n".join(f"- {o.text}" for o in a.observations) or "- (no structural eliminations)"
    inner_obs = _tag(f"factual_observations_operator_{n}", obs_lines)
    if not a.constrained:
        body = (
            f"Analysis for operator symbol `{a.op}`\n\n"
            f"{inner_obs}\n\n"
            f"Remaining candidates for `{a.op}`:\n- all operations (unconstrained by the examples)"
        )
        return _tag(f"operator_{n}_candidates", body)

    remaining = []
    for fam in a.candidate_families:
        offs = a.survivors[fam]
        mode_tag = _candidate_mode_label(fam, a)
        parts = [f"mode={mode_tag}"]
        if FAMILY_BY_NAME[fam].has_offset:
            parts.append(f"offsets={{{', '.join(_fmt_off(o) for o in offs)}}}")
        remaining.append(f"{fam} ({', '.join(parts)})")
    body = (
        f"Analysis for operator symbol `{a.op}`\n\n"
        f"{inner_obs}\n\n"
        f"Remaining candidates for `{a.op}`:\n- {', '.join(remaining)}"
    )
    return _tag(f"operator_{n}_candidates", body)


# ---------------------------------------------------------------------------
# Digit-mapping attempt blocks
# ---------------------------------------------------------------------------

def _render_mapping_next(
    step,
    order: list[str],
    running_allowed: dict[str, set[int]],
    just_removed: list[tuple[str, int]],
    base: int,
) -> str:
    """Render a subsequent mapping block with progressively reduced allowed sets.

    `just_removed`: (symbol, digit) pairs ruled out by the previous follow-up.
    `running_allowed`: per-symbol allowed sets AFTER those removals.
    """
    from src.cot_generator_v2.digit_constraint import _fmt_explicit

    fix = step.fixes_into  # the fix that PRODUCED this mapping
    if fix is None:
        change = ""
    elif fix.kind == "swap":
        a, b = fix.symbols
        change = f" (swapping {a} ↔ {b})"
    else:
        s = fix.symbols[0]
        change = f" (reassigning {s} to {step.mapping[s]})"

    lines: list[str] = []

    # Lead-in: name the digits ruled out by the previous follow-up
    if just_removed:
        ruled = ", ".join(f"{s} ≠ {d}" for s, d in just_removed)
        lines.append(
            f"Previous analysis ruled out: {ruled}. "
            f"Let's try this mapping{change}:\n"
        )
    else:
        lines.append(f"Following the previous observations, let's try this mapping{change}:\n")

    for s in order:
        d = step.mapping[s]
        aset = running_allowed.get(s, set(range(base)))
        aset_str = _fmt_explicit(aset)  # explicit: shows holes after progressive reduction
        lines.append(f"- {s} = {d}  (allowed: {aset_str})")

    if step.unused:
        lines.append(f"\nDigit candidates not yet used: {{{', '.join(map(str, step.unused))}}}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Conclusion helpers (Fix 8)
# ---------------------------------------------------------------------------

def _answer_format_observation(q: Equation, fam: str, examples: list[Equation]) -> str:
    """Fix 8 — observe answer width + sign position from example equations."""
    SIGNED_FAMILIES = {"sub_signed", "rsub_signed", "neg_absdiff"}
    same_op_eqs = [e for e in examples if e.op == q.op]

    # Part 1 — width / zero-padding
    if same_op_eqs:
        widths = [e.res_len for e in same_op_eqs]
        if len(set(widths)) == 1:
            w = widths[0]
            width_obs_lines = "\n".join(
                f"  - eq[{e.index}]: rhs has {e.res_len} symbol(s) → width = {e.res_len}"
                for e in same_op_eqs
            )
            width_obs = (
                f"- answer format (width): observing examples with `{q.op}`:\n"
                f"{width_obs_lines}\n"
                f"  All examples agree: answer width = {w}. "
                f"Zero-pad the result to {w} digit(s) if needed."
            )
        else:
            width_obs = (
                f"- answer format (width): examples with `{q.op}` show varying rhs lengths "
                f"({', '.join(str(w) for w in widths)}). "
                f"Using width from the question answer field ({q.res_len})."
            )
    else:
        width_obs = (
            f"- answer format (width): no example uses operator `{q.op}`. "
            f"Using width from the question answer field ({q.res_len}) as a best guess."
        )

    if fam not in SIGNED_FAMILIES:
        return width_obs

    # Part 2 — sign (only for families that can produce negative results)
    signed_eqs = [e for e in same_op_eqs if e.has_sign]
    if signed_eqs:
        e0 = signed_eqs[0]
        sign_pos = getattr(e0, "sign_pos", "postfix")
        sign_obs = (
            f"- answer format (sign): eq[{e0.index}] shows a signed result with "
            f"{sign_pos} convention. Applying {sign_pos} sign to the answer."
        )
    else:
        sign_obs = (
            f"- answer format (sign): no example with `{q.op}` shows a signed result. "
            "If the result is negative, guessing postfix convention "
            "(operator glyph appended after the magnitude)."
        )
    return f"{width_obs}\n{sign_obs}"


def _operand_symbol_display(examples: list[Equation], q: Equation) -> dict[str]:
    """Fix 10 — build the header operand-symbol list (constrained + unconstrained markers).

    Returns constrained symbols (first-appearance order from examples), followed by
    any question-operand symbols absent from examples suffixed with ' (unconstrained)'.
    Answer-only symbols (in q.rhs but not in q.operands or any example) are omitted —
    the reasoner cannot observe them at all before seeing the answer.
    """
    constrained = content_symbols(examples)
    cset = set(constrained)
    extra = [s for s in dict.fromkeys(q.operands) if s not in cset]
    
    return {
        "constrained" : list(constrained),
        "unconstrained" : [f"{s}," for s in extra]
    }   


def _decode_display(s0: str, s1: str, solution: dict[str, int],
                    base: int, mode: str, known: set[str]) -> str:
    """Fix 10 — produce the 'XY -> tok0,tok1 -> number' decode string.

    Shows 'unknown' for symbols absent from the known (example) set.
    When both symbols are known, shows the decoded integer.
    When one is unknown, shows a partial place-value representation:
      standard:      'tok0 tok1'  (tens digit first)
      little_endian: 'tok1 tok0'  (tens=d1 first, units=d0 second)
    """
    tok0 = str(solution[s0]) if s0 in known else "unknown"
    tok1 = str(solution[s1]) if s1 in known else "unknown"
    if s0 in known and s1 in known:
        number = str(decode_operand(s0, s1, solution, base, mode))
    elif mode == LITTLE_ENDIAN:
        # LE: value = d1 * base + d0  →  written as "d1 d0" (tens first)
        number = f"{tok1} {tok0}"
    else:
        # STD: value = d0 * base + d1  →  written as "d0 d1" (tens first)
        number = f"{tok0} {tok1}"
    return f"{s0}{s1} -> {tok0},{tok1} -> {number}"


def _apply_formula_str(fam: str, off: int, L: int, R: int) -> str:
    """Fix 10 — produce the formula with operand values substituted.

    Takes FAMILY_BY_NAME[fam].formula (e.g. '|a - b| + offset'), resolves the
    offset term, then substitutes a→L, b→R (safe: formula letters are standalone).
    """
    formula = FAMILY_BY_NAME[fam].formula
    if "+ offset" in formula:
        if off == 0:
            formula = formula.replace(" + offset", "")
        elif off > 0:
            formula = formula.replace("offset", str(off))
        else:
            formula = formula.replace("+ offset", f"- {abs(off)}")
    return formula.replace("a", str(L)).replace("b", str(R))


def _unknown_result_digits_note(digits_display: list[int],
                                solution: dict[str, int], known: set[str]) -> str:
    """Fix 10 — digit→symbol guess for result digits not covered by known symbols.

    'digits_display' is the post-mode-reversal digit list (display order).
    For each unique digit not assigned to any known (example) symbol, guesses
    the symbol via the inverse solution map.  Written in the reasoner's voice.
    """
    known_digits = {solution[s] for s in known}
    inv = {d: s for s, d in solution.items()}
    unknown = list(dict.fromkeys(d for d in digits_display if d not in known_digits))
    if not unknown:
        return ""
    d_list = ", ".join(str(d) for d in unknown)
    guesses = ", ".join(f"{d} = `{inv[d]}`" for d in unknown)
    return (
        f"- note: result digit(s) {d_list} never appear in any example equation, "
        "so their symbols cannot be deduced from the examples. "
        f"Making a best guess: {guesses}."
    )


def _answer_working(q, solution, fam, off, mode, base, answer, examples=None) -> str:
    """Show the decode → apply → encode steps that produce the boxed answer.

    Fix 10 (rework): distinguishes constrained (example) symbols from unconstrained
    question-operand and answer-only symbols.

    - decode: shows 'unknown' for unconstrained operand symbols.
    - guess:  states a best-guess digit for each unconstrained operand symbol.
    - apply:  shows the formula with values substituted (not the bare numeric result).
    - encode: shows digit encoding without the symbol re-mapping; notes zero padding.
    - note:   digit→symbol guess for result digits not covered by known symbols.

    Fix 8 (unchanged): inserts the answer-format observation between apply and encode.
    """
    lhs = f"{q.l0}{q.l1}{q.op}{q.r0}{q.r1}"
    head = f"Using the feasible mappings to answer the question {lhs} (`{mode}`):"
    if fam in ("concat_fwd", "concat_rev"):
        pat = "".join(concat_pattern(fam, q))
        return (f"{head}\n"
                f"- {fam} copies the operand symbols in written order: {lhs} -> {pat} = {answer}")

    known: set[str] = set(content_symbols(examples)) if examples is not None else set(solution)

    L = decode_operand(q.l0, q.l1, solution, base, mode)
    R = decode_operand(q.r0, q.r1, solution, base, mode)
    value = apply_family(fam, off, L, R, base)
    off_txt = f" (offset {_fmt_off(off)})" if FAMILY_BY_NAME[fam].has_offset else ""

    # Decode line
    dec_L = _decode_display(q.l0, q.l1, solution, base, mode, known)
    dec_R = _decode_display(q.r0, q.r1, solution, base, mode, known)
    decode_line = f"- decode: {dec_L}; {dec_R}"

    # Guess line for unconstrained operand symbols (omitted when all known)
    unc_ops = [s for s in dict.fromkeys(q.operands) if s not in known]
    guess_line = ""
    if unc_ops:
        syms_txt = ", ".join(f"`{s}`" for s in unc_ops)
        guesses = ", ".join(f"`{s}` = {solution[s]}" for s in unc_ops)
        guess_line = f"- unknown digit for symbol(s) {syms_txt}. Making a best guess: {guesses}"

    # Apply line: formula with values, not bare numeric result
    formula_expr = _apply_formula_str(fam, off, L, R)
    apply_line = f"- apply: {fam}({L}, {R}){off_txt} = {formula_expr}"

    # Fix 8: answer-format observation block (between apply and encode)
    fmt_obs = _answer_format_observation(q, fam, examples) if examples is not None else ""

    # Encode line: digit list without symbol re-mapping; zero-padding note
    digits_raw = int_to_base_digits(abs(value), base)
    if len(digits_raw) < q.res_len:
        digits_raw = [0] * (q.res_len - len(digits_raw)) + digits_raw
    nat = ",".join(str(d) for d in digits_raw)
    if mode == LITTLE_ENDIAN:
        digits_display = list(reversed(digits_raw))
        enc_txt = f"digits {nat} reversed to {','.join(str(d) for d in digits_display)}"
    else:
        digits_display = list(digits_raw)
        enc_txt = f"digits {nat}"
    zero_note = " (preserve the zero padding)" if 0 in digits_display else ""
    encode_line = f"- encode: |{value}| -> {enc_txt}{zero_note}"

    # Note line: digit→symbol guess for unknown result digits
    note_txt = _unknown_result_digits_note(digits_display, solution, known)

    parts = [head, decode_line]
    if guess_line:
        parts.append(guess_line)
    parts.append(apply_line)
    if fmt_obs:
        parts.append(fmt_obs)
    parts.append(encode_line)
    if note_txt:
        parts.append(note_txt)
    return "\n".join(parts)


def _render_conclusion(q, solution, program, mode, base, priority_order, answer,
                       examples=None) -> str:
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
        f"{_answer_working(q, solution, fam, off, mode, base, answer, examples=examples)}\n\n"
        f"The final answer is \\box{{{answer}}}\n\n"
        "Notice: this is one valid answer. Other combinations may also satisfy all constraints."
    )


# ---------------------------------------------------------------------------
# Pure-concat CoT (concise — no digit mapping needed)
# ---------------------------------------------------------------------------

def render_concat_cot(
    examples: list[Equation], q: Equation, fam: str,
) -> CotResult:
    """Render a short, tagged CoT for a pure-concat row (no digit mapping needed).

    Unified structure:
      <initial_reasoning>  — short puzzle layout (shared with arithmetic path)
      <concat_check>       — Outcome A: all relevant equations pass concat
      <conclusion>         — apply the pattern and emit \\box{answer}
    """
    operand_syms = _operand_symbol_display(examples, q)
    operator_syms = sorted({eq.op for eq in examples} | {q.op})

    ir = _render_initial_reasoning(examples, q, operand_syms, operator_syms)
    cc = render_concat_check_block(examples, q, fam)

    pat_syms = concat_pattern(fam, q)
    answer = "".join(pat_syms)
    pat_str = " ".join(pat_syms)
    conc = (
        f"Apply {fam} to the question lhs {q.l0}{q.l1}{q.op}{q.r0}{q.r1}:\n"
        f"  operands → {pat_str} → {answer}\n\n"
        f"The final answer is \\box{{{answer}}}"
    )

    # Fix 9c: drop <initial_reasoning> and <concat_check> outer tags; keep <conclusion>
    text = "\n\n".join([ir, cc, _tag("conclusion", conc)])
    return CotResult(text=text, answer=answer, mode="concat", program={q.op: (fam, 0)})


# ---------------------------------------------------------------------------
# Top-level arithmetic CoT
# ---------------------------------------------------------------------------

def render_cot(row: dict, cfg: CotGeneratorConfig, rounds: int | None = None,
               break_order: list[int] | None = None) -> CotResult:
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
    # Fix 10: constrained_syms drives the digit-mapping derivation (example symbols only).
    # The display list shown in the header marks unconstrained question operands and
    # omits answer-only symbols.
    constrained_syms = content_symbols(examples)
    operand_syms_display = _operand_symbol_display(examples, q)
    operator_syms = sorted({eq.op for eq in examples} | {q.op})

    # Anchor to the solver's actual program so the conclusion reproduces the
    # labelled answer (puzzles are underdetermined).
    mode = "little_endian" if "little" in str(row["solver_strategy"]).lower() else "standard"
    solver_prog = {op: resolve_registry_op(name) for op, name in om.items()}

    # Build a full protect map: every gold op's family is shielded from both the
    # sign heuristic drop (sub_signed / rsub_signed) and the redundancy drop.
    gold_protect = {op: fam for op, (fam, _off) in solver_prog.items()}
    analyses = analyze_operators(examples, q.op, base, protect=gold_protect)
    apply_redundancies(analyses, protect=gold_protect)

    program = solution_program(analyses, solution, base, mode, solver_prog)

    # Symbol order: by raw occurrence count (highest first) — simpler pool-only path.
    prio = compute_digit_priority(analyses, examples, constrained_syms,
                                  "occurrences", cfg.random_seed, problem_id)
    order = prio.order

    # Backward perturbation chain — built before picks so picks use the first-attempt
    # mapping (chain.steps[0]) as the oracle rather than the final solution.
    chain = build_chain(
        solution, constrained_syms, examples, program, base,
        cfg.perturbation_rounds if rounds is None else rounds,
        cfg.random_seed, problem_id, mode,
        break_order=break_order,
    )
    for i, st in enumerate(chain.steps):
        st.fixes_into = None if i == 0 else chain.steps[i - 1].fixes[0]

    num_rounds = len(chain.steps) - 1  # realized perturbation count

    # Full-pool allowed map: no constraint analysis, each symbol can take any digit.
    # picks use chain.steps[0].mapping as oracle so digit_mapping_1 matches the
    # first attempt that the analysis/follow_up blocks actually evaluate.
    full_allowed = {s: set(range(base)) for s in order}
    first_attempt_mapping = chain.steps[0].mapping
    picks = picking_sequence(full_allowed, order, first_attempt_mapping, base)

    parts: list[str] = []

    # 1. Short shared initial reasoning with budget sentence (Fix 9a, 9c: no outer tag)
    parts.append(_render_initial_reasoning(examples, q, operand_syms_display, operator_syms))

    # 2. Concat check — Outcome B (Fix 9c: no outer tag)
    parts.append(render_concat_check_block(examples, q, fam=None))

    # 3. Encoding explanation (Fix 9c: no outer tag)
    parts.append(_render_encoding_explanation())

    # 4. Available operations (Fix 9c: no outer tag)
    parts.append(
        "First, let's identify the possible operators and discard infeasible ones based "
        "solely on the form and structure of the example equations:\n\n"
        + _available_operations())

    # 5. Per-operator structural analysis (Fix 9c: drop operator_N_candidates and
    #    factual_observations tags — render text only)
    parts.append(
        "Let's analyze the operator symbols one by one and make factual observations based "
        "on the form of the equations:")
    for n, a in enumerate(analyses, start=1):
        # Strip the outer tag, keep the body text
        block = _render_operator_block(n, a, q)
        # Remove outer <operator_N_candidates> tag (keep inner factual_obs for readability)
        inner = block
        open_tag = f"<operator_{n}_candidates>"
        close_tag = f"</operator_{n}_candidates>"
        if inner.startswith(open_tag) and inner.endswith(close_tag):
            inner = inner[len(open_tag):-len(close_tag)].strip()
        parts.append(inner)

    # 6. Transition sentence (no outer tag; detailed reasoning lives inside digit_mapping_1)
    parts.append("Now, let's derive the digit mapping from the example equations.")

    # 7. Perturbation chain steps (Fix 9b: append running \boxed{}; Fix 9c: keep 4 core tags)
    #    running_allowed tracks per-symbol options; shrinks after each follow-up reveals bad digits.
    prev_map = None
    prev_attempt: AttemptAnalysis | None = None
    answer = predict_answer(q, *program[q.op], solution, base, mode)
    if num_rounds > 0:
        parts.append(
            f"We will attempt up to {num_rounds + 1} digit mapping(s). After each attempt "
            "we record the current best candidate. If no fully feasible mapping is found "
            "within the budget, we conclude with the best candidate found so far."
        )

    running_allowed: dict[str, set[int]] = {s: set(range(base)) for s in order}
    just_removed: list[tuple[str, int]] = []

    for i, st in enumerate(chain.steps):
        n = st.index
        if i == 0:
            # Occurrence-ordered pool picking (no per-equation constraint analysis).
            block_body = render_first_mapping_block_occurrences(prio.text, picks, base)
        else:
            # Subsequent attempts: show progressively narrowed allowed sets
            block_body = _render_mapping_next(st, order, running_allowed, just_removed, base)
        parts.append(_tag(f"digit_mapping_{n}", block_body))

        attempt = analyze_attempt(n, st.mapping, prev_map, analyses, prev_attempt, base, program)
        parts.append(_tag(f"digit_mapping_analysis_{n}", attempt.text))
        follow = make_tell(st, examples, solution, program, base, mode, cfg.tell_mode)

        # Fix 9b: append running \boxed{} to every follow-up block.
        # Broken mappings may not cover all result digits → catch and use placeholder.
        try:
            step_answer = predict_answer(q, *program[q.op], st.mapping, base, mode)
        except (KeyError, ValueError):
            step_answer = "?"
        if attempt.is_solution:
            follow = follow + f"\nAll equations satisfied. Answer confirmed: \\boxed{{{step_answer}}}"
        else:
            follow = follow + f"\nCurrent best candidate: \\boxed{{{step_answer}}}"
        parts.append(_tag(f"digit_mapping_analysis_follow_up_{n}", follow))

        # Reduce running_allowed for the next mapping block based on this step's fixes.
        running_allowed, just_removed = reduce_allowed_from_followup(running_allowed, st)

        prev_map = st.mapping
        prev_attempt = attempt

    # 8. Conclusion (keep <conclusion> tag)
    parts.append(_tag("conclusion",
                      _render_conclusion(q, solution, program, mode, base, order,
                                         answer, examples=examples)))

    return CotResult(text="\n\n".join(parts), answer=answer, mode=mode, program=program,
                     num_perturbations=num_rounds)


def render_guess_conclusion(mapping: dict, gold_answer: str, attempts_used: int) -> str:
    """Fix 9 / Aug 4: render a budget-exhausted conclusion block."""
    return (
        f"Budget of {attempts_used} attempt(s) used. No fully feasible mapping confirmed "
        f"across all equations.\nReturning best candidate found: \\boxed{{{gold_answer}}}"
    )


def _selftest() -> None:
    import pandas as pd
    df = pd.read_parquet("inputs/data/v2/only_cryptharithm_solved_k_1.parquet")
    row = df.iloc[0].to_dict()
    cfg = CotGeneratorConfig(digit_priority_mode="most_constrained", tell_mode="algebraic",
                            perturbation_rounds=2, random_seed=0)
    res = render_cot(row, cfg)
    assert res.answer == row["solver_answer"], (res.answer, row["solver_answer"])
    # Fix 9c: structural tags removed, keep the 4 core ones + conclusion
    for tag in ("digit_mapping_1", "digit_mapping_analysis_1",
                "digit_mapping_analysis_follow_up_1", "conclusion"):
        assert f"<{tag}>" in res.text, f"missing <{tag}>"
    # Fix 9b: running \boxed{} in every follow-up
    assert "\\boxed{" in res.text, "missing running \\boxed{}"
    # Fix 9c: structural tags should NOT appear as outer wrappers
    assert "<initial_reasoning>" not in res.text, "initial_reasoning tag should be removed"
    assert "<available_operations>" not in res.text, "available_operations tag should be removed"
    # Fix 4: concat check text present
    assert "Concat does not apply" in res.text, "missing concat ruling-out text"
    # Fix 9a: budget sentence
    assert "We will attempt up to" in res.text, "missing budget sentence"
    print("render self-test PASS  answer =", res.answer)


if __name__ == "__main__":
    _selftest()
