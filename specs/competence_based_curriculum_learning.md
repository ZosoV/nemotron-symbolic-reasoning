# Competence-Based Curriculum Learning — Steps

A step-by-step summary of the competence-based curriculum (Platanios et al.), adapted to a systems-of-equations task where difficulty = number of example equations.

1. **Assign a raw difficulty score** to every training example. Here, `d̂(x) = n` (number of equations).

2. **Normalize difficulty to [0, 1]** using the empirical CDF — each example's difficulty becomes the fraction of the dataset that is *easier* than it, making the scale dataset-relative rather than raw.

3. **Pick a competence function** `c(t)` that grows from a small initial value `c₀ ≈ 0.01` to `1` at a chosen step `T`:
   - **Sqrt** (more time on harder sizes): `c(t) = min(1, √(t·(1−c₀²)/T + c₀²))`
   - **Linear** (even spacing): `c(t) = min(1, t·(1−c₀)/T + c₀)`

4. **Set the hyperparameters** `c₀` (starting competence) and `T` (step at which the full dataset unlocks).

5. **At each training step `t`, compute `c(t)`** from the competence function.

6. **Filter the available pool** to examples with normalized difficulty `≤ c(t)`.

7. **Sample the minibatch uniformly** from that filtered pool and take the usual gradient step.

8. **Repeat steps 5–7**, letting competence rise so harder systems progressively unlock while easier ones remain in the pool (built-in replay).

9. **Stop the curriculum at `t = T`**, where `c(t) = 1` and the model trains on the full dataset; continue normal training to convergence.

10. **Evaluate on held-out difficulties**, including sizes beyond the hardest training bucket, to verify the curriculum produced a *generalizing procedure* rather than a per-size fit.