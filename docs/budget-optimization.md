# Budget optimization

Once the model knows each channel's diminishing-returns curve, the natural
business question is: **given a fixed weekly budget, how should we split it?**

## Approach

Rather than the library's built-in optimiser (which works in the model's
internally-scaled space and behaves poorly through a save/load round-trip with
the classic `MMM`), this project reconstructs each channel's **steady-state
response curve in original sales units** from the posterior and optimises
allocation directly. The result is transparent and easy to reason about.

For a sustained weekly spend `s` (a constant input passes through normalised
adstock unchanged):

$$
R_c(s) = \max(\text{sales}) \cdot \beta_c \cdot \text{sat}\!\left(\frac{s}{\max(\text{spend}_c)};\, \lambda_c\right)
$$

where the `max` terms undo the model's max-abs scaling. See
[`ResponseCurve`][bmmm.model.budget.ResponseCurve].

## The optimisation

[`optimize_budget`][bmmm.model.budget.optimize_budget] maximises total weekly
response subject to a fixed-budget constraint:

$$
\max_{s_1,\dots,s_C} \sum_c R_c(s_c) \quad\text{s.t.}\quad \sum_c s_c = B,\; s_c \ge 0
$$

Each `R_c` is concave (saturating), so the objective is concave with a linear
constraint — a well-behaved problem with a **unique optimum**, solved with SciPy
SLSQP. Economically, the optimum equalises **marginal ROI** across channels.

## Example

Re-running at the historical budget shows it is already near-optimal (a realistic
finding). Raising the budget by ~20 % reveals where the extra money should go:

| Channel | Current | Optimal (+20 % budget) | Change |
|---|---|---|---|
| tv | 647 | 876 | **+35 %** |
| social | 394 | 450 | +14 % |
| search | 271 | 274 | +1 % |

**Response uplift: +14.5 %.** Most of the extra budget flows to TV, which is the
least saturated channel at current spend; search is already near its ceiling, so
it barely moves.

```bash
uv run bmmm optimize-budget --budget 1600
```

The same logic powers the [API](usage-api.md) `/optimize-budget` endpoint and the
interactive dashboard's scenario planner.
