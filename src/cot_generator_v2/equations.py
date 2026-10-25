"""Self-contained equation parsing for the CoT generator.

An encrypted equation has the form  `l0 l1 op r0 r1 = rhs`  where l0,l1,r0,r1
are content (digit) symbols, op is an operator symbol, and rhs is the result
magnitude, optionally carrying a sign affix equal to the operator glyph
(prefix = standard, postfix = little_endian).

This re-implements the parts of src/parsing.py the generator needs, so the
package has no dependency on the solver's parser. The structured shape matches
the `equations` parquet column, so a row's pre-parsed equations can be used
directly as well.

Run (self-test):
    uv run python src/cot_generator/equations.py
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Equation:
    """A parsed equation.

    l0, l1  — left operand symbols   (L decodes from these two)
    op      — operator glyph
    r0, r1  — right operand symbols  (R decodes from these two)
    rhs     — result magnitude symbols (sign affix already stripped)
    has_sign — True when the result carried a sign affix (negative result)
    kind    — 'example' | 'to_predict'
    index   — 1-based position among example equations (0 for to_predict)
    raw     — the original "lhs = rhs" string, for verbatim display
    """
    l0: str
    l1: str
    op: str
    r0: str
    r1: str
    rhs: tuple[str, ...]
    has_sign: bool
    sign_pos: str | None         # 'prefix' | 'postfix' | None
    kind: str
    index: int
    raw: str

    @property
    def res_len(self) -> int:
        return len(self.rhs)

    @property
    def operands(self) -> tuple[str, str, str, str]:
        return (self.l0, self.l1, self.r0, self.r1)

    @property
    def symbols(self) -> tuple[str, ...]:
        """All content symbols touched by this equation (operands + rhs)."""
        return self.operands + self.rhs

    def self_equal(self) -> bool:
        """True when both operands are the same glyph pair (L == R structurally)."""
        return self.l0 == self.r0 and self.l1 == self.r1


def parse_equation_str(eq_str: str, kind: str, index: int) -> Equation:
    """Parse one "lhs = rhs" string into an Equation."""
    lhs_str, rhs_str = eq_str.split(" = ", 1)
    lhs_str, rhs_str = lhs_str.strip(), rhs_str.strip()
    if len(lhs_str) != 5:
        raise ValueError(f"LHS must be exactly 5 characters, got {lhs_str!r}")

    op = lhs_str[2]
    chars = list(rhs_str)
    has_sign = False
    sign_pos: str | None = None
    # Detect trailing sign first, guarding against a length-1 leading-sign rhs.
    if len(chars) > 1 and chars[-1] == op and chars[0] != op:
        has_sign, sign_pos, chars = True, "postfix", chars[:-1]
    elif chars and chars[0] == op:
        has_sign, sign_pos, chars = True, "prefix", chars[1:]

    return Equation(
        l0=lhs_str[0], l1=lhs_str[1], op=op, r0=lhs_str[3], r1=lhs_str[4],
        rhs=tuple(chars), has_sign=has_sign, sign_pos=sign_pos,
        kind=kind, index=index, raw=f"{lhs_str} = {rhs_str}",
    )


def parse_prompt(prompt: str, answer: str) -> tuple[list[Equation], Equation]:
    """Parse a puzzle prompt into (example equations, to_predict equation).

    The question line ("Now, determine the result for: <lhs>") is combined with
    the gold `answer` to form the to_predict equation.
    """
    examples: list[Equation] = []
    question_lhs: str | None = None
    for line in prompt.split("\n"):
        line = line.strip()
        if line.startswith("Now"):
            question_lhs = line.split(": ", 1)[1].strip()
        elif " = " in line:
            examples.append(parse_equation_str(line, "example", len(examples) + 1))
    if question_lhs is None:
        raise ValueError("prompt has no 'Now, determine...' question line")
    to_predict = parse_equation_str(f"{question_lhs} = {answer}", "to_predict", 0)
    return examples, to_predict


def content_symbols(equations: list[Equation]) -> list[str]:
    """Unique content symbols in first-appearance order (op glyphs excluded)."""
    op_chars = {eq.op for eq in equations}
    seen: dict[str, None] = {}
    for eq in equations:
        for ch in eq.symbols:
            if ch not in op_chars and ch not in seen:
                seen[ch] = None
    return list(seen.keys())


def operator_groups(examples: list[Equation]) -> dict[str, list[Equation]]:
    """Group example equations by operator glyph, preserving first-seen order."""
    groups: dict[str, list[Equation]] = {}
    for eq in examples:
        groups.setdefault(eq.op, []).append(eq)
    return groups


def occurrences(symbol: str, equations: list[Equation]) -> int:
    """Total count of `symbol` across all operand + rhs slots of `equations`."""
    n = 0
    for eq in equations:
        n += sum(1 for s in eq.symbols if s == symbol)
    return n


def _selftest() -> None:
    prompt = (
        "In Alice's Wonderland, a secret set of transformation rules is applied "
        "to equations. Below are a few examples:\n"
        "[[-!' = @&\n`!*\\& = '@'{\n\\'-!` = \\\\\n\n"
        "Now, determine the result for: `!*[{"
    )
    examples, q = parse_prompt(prompt, "'\"[`")
    assert len(examples) == 3
    assert q.op == "*" and q.kind == "to_predict"
    e_minus = examples[0]
    assert e_minus.op == "-" and e_minus.rhs == ("@", "&") and not e_minus.has_sign
    e3 = examples[2]  # \'-!` = \\
    assert e3.l0 == "\\" and e3.rhs == ("\\", "\\")
    groups = operator_groups(examples)
    assert set(groups) == {"-", "*"} and len(groups["-"]) == 2 and len(groups["*"]) == 1
    syms = content_symbols(examples)
    assert "*" not in syms and "-" not in syms
    assert occurrences("!", examples) == 3  # once in each of the three example eqs
    print("equations self-test PASS")


if __name__ == "__main__":
    _selftest()
