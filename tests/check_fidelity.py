"""Fidelity check: compare a synthetic augmented dataset against the real source.

Computes every statistic enforced by the v3 synthetic generator and prints a
PASS/FAIL table.  Exit code is 0 if all checks pass, 1 otherwise.

Usage:
    uv run python tests/check_fidelity.py [SYNTH_PARQUET] [--source SOURCE_PARQUET]

Defaults:
    SYNTH_PARQUET  inputs/data/v3/synthetic_v3_1k_canonical.parquet
    --source       inputs/data/v2/only_cryptharithm_solved_k_1.parquet
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_SOURCE_DEFAULT = _ROOT / "inputs" / "data" / "v2" / "only_cryptharithm_solved_k_1.parquet"
_SYNTH_DEFAULT = _ROOT / "inputs" / "data" / "v3" / "synthetic_v3_1k_canonical.parquet"

# ── Tolerances ──────────────────────────────────────────────────────────────
# Each tolerance is an absolute percentage-point delta (e.g. 5 → ±5 pp).
_TOL_STRATEGY = 5        # strategy distribution per bucket
_TOL_N_OPS = 5          # distinct operators per problem per bucket
_TOL_N_OSYM = 7         # unique operand symbols per problem per bucket
_TOL_MAG_LEN = 6        # result magnitude length per bucket
_TOL_SIGN_RATE = 8      # overall sign rate (known ~6 pp gap is accepted)
_TOL_SIGN_POS = 3       # prefix/postfix split
_TOL_ZERO_PAD = 1.5     # zero-padding rate
_TOL_CONCAT_NC = 15     # n_concat per bucket (small counts → wider tolerance)

# Forbidden operand / operator symbols (never appear in real data)
_FORBIDDEN_SYMS = {",", ".", ";", "_", "~"}


# ── Stats extraction ─────────────────────────────────────────────────────────

def _extract_stats(df, label: str) -> dict:
    """Return a dict of computed statistics for one dataset."""
    from src.encoding import _int_to_base_digits, _is_reversed_digit_mode
    from src.op_index import CONCAT_OPS
    from src.operations import OPERATIONS

    n_ops_c: Counter[int] = Counter()
    n_osym_c: Counter[int] = Counter()
    mag_len_c: Counter[int] = Counter()
    strategy_c: Counter[str] = Counter()
    sign_total = 0
    sign_count = 0
    sign_pos_c: Counter[str] = Counter()
    zp_total = 0
    zp_count = 0
    concat_nc_c: Counter[int] = Counter()
    sign_demo_total = 0
    sign_demo_count = 0
    pad_demo_total = 0
    pad_demo_count = 0
    all_operand_syms: set[str] = set()
    all_operator_syms: set[str] = set()
    pc_issues: list[str] = []  # pure_concat structural violations

    for _, row in df.iterrows():
        om_raw = row.get("ops_mapping")
        om: dict[str, str] = json.loads(om_raw) if isinstance(om_raw, str) else (om_raw if isinstance(om_raw, dict) else {})
        dm_raw = row.get("digits_mapping")
        dm: dict[str, int] = json.loads(dm_raw) if isinstance(dm_raw, str) else (dm_raw if isinstance(dm_raw, dict) else {})
        eqs_raw = row.get("equations")
        eqs: list[dict] = json.loads(eqs_raw) if isinstance(eqs_raw, str) else (eqs_raw or [])
        strategy = str(row.get("solver_strategy", ""))
        strategy_c[strategy] += 1
        le = _is_reversed_digit_mode("little_endian" if strategy == "little_endian" else "standard")
        is_arith = strategy in ("standard", "little_endian")
        is_pc = strategy == "pure_concat"

        prob_glyphs: set[str] = set()
        prob_osyms: set[str] = set()
        pc_concat_ex = 0
        eq_infos: list[tuple] = []  # (type, op_name, signed, padded)

        for eq in eqs:
            lhs = eq.get("lhs", {})
            rhs = eq.get("rhs", {})
            op_char = lhs.get("op", "")
            op_name = om.get(op_char, "")

            if op_char:
                prob_glyphs.add(op_char)
                all_operator_syms.add(op_char)
            for pos in ("l0", "l1", "r0", "r1"):
                s = lhs.get(pos, "")
                if s:
                    prob_osyms.add(s)
                    all_operand_syms.add(s)
            sign_sym = rhs.get("op")
            for s in rhs.get("actual_rhs", []):
                if s and s != sign_sym:
                    prob_osyms.add(s)
                    all_operand_syms.add(s)

            signed = sign_sym is not None
            if signed:
                sp = rhs.get("sign_pos") or "prefix"
                sign_pos_c[str(sp)] += 1

            if is_pc and eq.get("type") == "example" and op_name in CONCAT_OPS:
                pc_concat_ex += 1

            padded = False
            if is_arith and dm and op_name and op_name not in CONCAT_OPS:
                try:
                    l0d, l1d = dm[lhs["l0"]], dm[lhs["l1"]]
                    r0d, r1d = dm[lhs["r0"]], dm[lhs["r1"]]
                    L = l1d * 10 + l0d if le else l0d * 10 + l1d
                    R = r1d * 10 + r0d if le else r0d * 10 + r1d
                    fn = OPERATIONS.get(op_name)
                    if fn:
                        res = fn(L, R)
                        if isinstance(res, (int, float)):
                            nat_len = len(_int_to_base_digits(abs(int(res)), 10))
                            ar_len = len(rhs.get("actual_rhs", []))
                            mag_len_c[ar_len] += 1
                            sign_total += 1
                            if signed:
                                sign_count += 1
                            zp_total += 1
                            if ar_len > nat_len:
                                padded = True
                                zp_count += 1
                except (KeyError, TypeError, ZeroDivisionError):
                    pass

            eq_infos.append((eq.get("type"), op_name, signed, padded))

        n_ops_c[len(prob_glyphs)] += 1
        n_osym_c[len(prob_osyms)] += 1

        if is_pc:
            if pc_concat_ex > 0:
                concat_nc_c[pc_concat_ex] += 1
            # Structural checks
            dm_val = row.get("digits_mapping")
            if dm_val not in (None, "null", "nan") and str(dm_val) not in ("None", "nan", "null", "{}"):
                try:
                    parsed = json.loads(dm_val) if isinstance(dm_val, str) else dm_val
                    if parsed:
                        pc_issues.append(f"non-null digits_mapping in row {row.get('problem_id','?')}: {parsed}")
                except Exception:
                    pass
            concat_glyphs_in_om = [g for g, n in om.items() if n in CONCAT_OPS]
            non_concat_in_om = [g for g, n in om.items() if n not in CONCAT_OPS]
            if non_concat_in_om:
                pc_issues.append(f"non-concat op in ops_mapping for pc row {row.get('problem_id','?')}: {non_concat_in_om}")
            if not concat_glyphs_in_om:
                pc_issues.append(f"no concat glyph in ops_mapping for pc row {row.get('problem_id','?')}")

        # Demo tracking (arithmetic only)
        if is_arith and dm:
            tps = [(s, p, op) for (t, op, s, p) in eq_infos if t == "to_predict"]
            exs = [(s, p, op) for (t, op, s, p) in eq_infos if t == "example"]
            if tps:
                tp_s, tp_p, tp_op = tps[0]
                if tp_s:
                    sign_demo_total += 1
                    if any(s and op == tp_op for s, p, op in exs):
                        sign_demo_count += 1
                if tp_p:
                    pad_demo_total += 1
                    if any(p and op == tp_op for s, p, op in exs):
                        pad_demo_count += 1

    total = sum(strategy_c.values())

    return {
        "label": label,
        "n_rows": len(df),
        "strategy": {k: v / total for k, v in strategy_c.items()},
        "n_ops": dict(n_ops_c),
        "n_osym": dict(n_osym_c),
        "mag_len": dict(mag_len_c),
        "sign_rate": sign_count / max(sign_total, 1),
        "sign_pos": dict(sign_pos_c),
        "zero_pad": zp_count / max(zp_total, 1),
        "concat_nc": dict(concat_nc_c),
        "sign_demo": (sign_demo_count, sign_demo_total),
        "pad_demo": (pad_demo_count, pad_demo_total),
        "all_operand_syms": all_operand_syms,
        "all_operator_syms": all_operator_syms,
        "pc_issues": pc_issues,
        "n_pc": strategy_c.get("pure_concat", 0),
    }


# ── Check helpers ─────────────────────────────────────────────────────────────

def _dist_to_pct(counter: dict[int, int]) -> dict[int, float]:
    total = sum(counter.values())
    if total == 0:
        return {}
    return {k: v / total * 100 for k, v in sorted(counter.items())}


def _check_dist(name: str, real_c: dict, synth_c: dict, tol: float, results: list[tuple]):
    real_pct = _dist_to_pct(real_c)
    synth_pct = _dist_to_pct(synth_c)
    all_keys = sorted(set(real_pct) | set(synth_pct))
    for k in all_keys:
        r = real_pct.get(k, 0.0)
        s = synth_pct.get(k, 0.0)
        delta = abs(r - s)
        ok = delta <= tol
        results.append((ok, f"{name}[{k}]", f"{r:.1f}%", f"{s:.1f}%", f"Δ={delta:.1f}pp (tol ±{tol}pp)"))


def _check_scalar(name: str, real_v: float, synth_v: float, tol: float, results: list[tuple], unit: str = "%"):
    delta = abs(real_v - synth_v)
    ok = delta <= tol
    results.append((ok, name, f"{real_v:.2f}{unit}", f"{synth_v:.2f}{unit}", f"Δ={delta:.2f} (tol ±{tol})"))


# ── Main ──────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fidelity check: synth vs real dataset.")
    parser.add_argument("synth", nargs="?", default=str(_SYNTH_DEFAULT), help="Synthetic parquet path")
    parser.add_argument("--source", default=str(_SOURCE_DEFAULT), help="Real source parquet path")
    args = parser.parse_args(argv)

    import pandas as pd

    real_path = Path(args.source)
    synth_path = Path(args.synth)

    if not real_path.exists():
        print(f"[ERROR] Source not found: {real_path}", file=sys.stderr)
        return 1
    if not synth_path.exists():
        print(f"[ERROR] Synth not found: {synth_path}", file=sys.stderr)
        return 1

    real_df = pd.read_parquet(real_path)
    if "solver_correct" in real_df.columns:
        real_df = real_df[real_df["solver_correct"] == True].copy()
    synth_df = pd.read_parquet(synth_path)

    print(f"\nSource : {real_path.name}  ({len(real_df)} rows)")
    print(f"Synth  : {synth_path.name}  ({len(synth_df)} rows)")

    real = _extract_stats(real_df, "real")
    synth = _extract_stats(synth_df, "synth")

    results: list[tuple] = []  # (ok, name, real_val, synth_val, note)

    # ── 1. Strategy distribution ──────────────────────────────────────────
    for strat in ("standard", "little_endian", "pure_concat"):
        r = real["strategy"].get(strat, 0) * 100
        s = synth["strategy"].get(strat, 0) * 100
        delta = abs(r - s)
        ok = delta <= _TOL_STRATEGY
        results.append((ok, f"strategy[{strat}]", f"{r:.1f}%", f"{s:.1f}%", f"Δ={delta:.1f}pp (tol ±{_TOL_STRATEGY}pp)"))

    # ── 2. Operator count per problem ──────────────────────────────────────
    _check_dist("n_ops", real["n_ops"], synth["n_ops"], _TOL_N_OPS, results)

    # ── 3. Unique operand symbols per problem ──────────────────────────────
    real_osym = {k: v for k, v in real["n_osym"].items() if k <= 10}
    synth_osym = {k: v for k, v in synth["n_osym"].items() if k <= 10}
    _check_dist("n_osym", real_osym, synth_osym, _TOL_N_OSYM, results)

    # ── 3b. Hard cap: no synth problem should exceed 10 operand syms (arith) ──
    arith_over10 = sum(v for k, v in synth["n_osym"].items()
                       if k > 10 and k <= 30)  # >30 is pure_concat which may vary
    results.append((
        arith_over10 == 0,
        "n_osym_cap≤10 (arith)",
        "0 violations",
        f"{arith_over10} violations",
        "hard cap — any violation is a FAIL",
    ))

    # ── 4. Result magnitude length ─────────────────────────────────────────
    _check_dist("mag_len", real["mag_len"], synth["mag_len"], _TOL_MAG_LEN, results)

    # ── 5. Sign rate ───────────────────────────────────────────────────────
    _check_scalar("sign_rate", real["sign_rate"] * 100, synth["sign_rate"] * 100, _TOL_SIGN_RATE, results)

    # ── 6. Sign position (prefix vs postfix) ──────────────────────────────
    total_real_sign = sum(real["sign_pos"].values()) or 1
    total_synth_sign = sum(synth["sign_pos"].values()) or 1
    for pos in ("prefix", "postfix"):
        r = real["sign_pos"].get(pos, 0) / total_real_sign * 100
        s = synth["sign_pos"].get(pos, 0) / total_synth_sign * 100
        delta = abs(r - s)
        ok = delta <= _TOL_SIGN_POS
        results.append((ok, f"sign_pos[{pos}]", f"{r:.1f}%", f"{s:.1f}%", f"Δ={delta:.1f}pp (tol ±{_TOL_SIGN_POS}pp)"))

    # ── 7. Zero padding ────────────────────────────────────────────────────
    _check_scalar("zero_pad_rate", real["zero_pad"] * 100, synth["zero_pad"] * 100, _TOL_ZERO_PAD, results)

    # ── 8. Pure-concat n_concat distribution ──────────────────────────────
    _check_dist("concat_n_concat", real["concat_nc"], synth["concat_nc"], _TOL_CONCAT_NC, results)

    # ── 9. Symbol pool: no forbidden symbols ──────────────────────────────
    bad_op = synth["all_operator_syms"] & _FORBIDDEN_SYMS
    bad_operand = synth["all_operand_syms"] & _FORBIDDEN_SYMS
    results.append((len(bad_op) == 0, "forbidden_operator_syms", "∅", str(bad_op or "∅"), "must be empty"))
    results.append((len(bad_operand) == 0, "forbidden_operand_syms", "∅", str(bad_operand or "∅"), "must be empty"))

    # ── 10. Pure-concat structural integrity ──────────────────────────────
    results.append((
        len(synth["pc_issues"]) == 0,
        "pc_structural (digits_mapping=None, ops_mapping=concat_only)",
        "0 issues",
        f"{len(synth['pc_issues'])} issue(s)",
        "; ".join(synth["pc_issues"][:3]) or "OK",
    ))

    # ── Print table ────────────────────────────────────────────────────────
    print()
    w = (42, 14, 14, 30)
    header = f"{'Check':<{w[0]}} {'Real':>{w[1]}} {'Synth':>{w[2]}} {'Note':<{w[3]}}"
    print(header)
    print("─" * sum(w))

    n_pass = 0
    n_fail = 0
    for ok, name, real_v, synth_v, note in results:
        mark = "✓" if ok else "✗"
        print(f"{mark} {name:<{w[0]-2}} {real_v:>{w[1]}} {synth_v:>{w[2]}} {note:<{w[3]}}")
        if ok:
            n_pass += 1
        else:
            n_fail += 1

    print("─" * sum(w))
    print(f"\n{'PASSED' if n_fail == 0 else 'FAILED'}  {n_pass}/{n_pass+n_fail} checks passed")

    if n_fail > 0:
        print("\nFailed checks:")
        for ok, name, real_v, synth_v, note in results:
            if not ok:
                print(f"  ✗ {name}: real={real_v}  synth={synth_v}  ({note})")

    # ── Demo rates (informational only, high variance at small n) ──────────
    sd_n, sd_d = synth["sign_demo"]
    pd_n, pd_d = synth["pad_demo"]
    rd_n, rd_d = real["sign_demo"]
    rpd_n, rpd_d = real["pad_demo"]
    print(f"\n── Informational (high variance, not checked) ──")
    print(f"  sign_demo:  real={rd_n}/{rd_d} ({rd_n/max(rd_d,1)*100:.0f}%)  "
          f"synth={sd_n}/{sd_d} ({sd_n/max(sd_d,1)*100:.0f}%)  target≈54%")
    print(f"  pad_demo:   real={rpd_n}/{rpd_d} ({rpd_n/max(rpd_d,1)*100:.0f}%)  "
          f"synth={pd_n}/{pd_d} ({pd_n/max(pd_d,1)*100:.0f}%)  target≈35%")

    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
