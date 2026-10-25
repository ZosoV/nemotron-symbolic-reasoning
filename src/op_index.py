"""Lockstep mapping between operations.py op names and Rust apply() indices.

CRITICAL: The integer indices here must match the match arms in
rust/alice_sovler_helper/src/lib.rs apply() exactly. Any drift causes
silent wrong answers.
"""

# op_name -> Rust apply() index
OP_NAME_TO_IDX: dict[str, int] = {
    "add":          0,
    "absdiff":      3,
    "neg_absdiff":  4,
    "mul":          5,
    "gcd":          6,
    "lcm":          7,
    "fdiv":         8,
    "mod":          10,
    "rmod":         11,
    "add_m1":       14,
    "add_p1":       15,
    "mul_m1":       16,
    "mul_p1":       17,
    "absdiff_m1":   18,
    "absdiff_p1":   19,
    "add_p2":       25,
    "absdiff_m2":   28,
    "a2_plus_b":    38,
    "sub_signed":   43,
    "rsub_signed":  44,
    "concat_fwd":   45,
    "concat_rev":   46,
}

OP_IDX_TO_NAME: dict[int, str] = {v: k for k, v in OP_NAME_TO_IDX.items()}

# Ops whose result may be negative; the sign is encoded as a prefix on the RHS.
SIGNED_OPS: frozenset[str] = frozenset({"sub_signed", "rsub_signed"})

# Ops resolved structurally (no arithmetic); always unsigned, always res_len == 4.
CONCAT_OPS: frozenset[str] = frozenset({"concat_fwd", "concat_rev"})
