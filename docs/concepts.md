# Concepts

This page explains the general ideas behind the project: first what a Marketing
Mix Model is, then how a Bayesian model is fitted. It is theory, not specific to
this project. For what this project built and the numbers it produced, see
[Parameter recovery](parameter-recovery.md).

## What an MMM is

A Marketing Mix Model explains a business outcome, usually sales, as a baseline
plus the effect of each marketing channel:

$$
\text{sales}_t = \text{baseline}_t + \sum_{c} \text{contribution}_{c,t} + \varepsilon_t
$$

It is used to answer two questions: how much did each channel add to sales, and
how should the budget be split. Two things make advertising spend behave
differently from an ordinary regressor, and an MMM models both: spend keeps
working after the week it runs (adstock), and each extra unit of spend adds less
than the previous one (saturation).

## Adstock (carry-over)

Advertising in one week still affects later weeks. Geometric adstock models this
as a memory that fades by a factor `alpha` each week, where `alpha` is between 0
and 1. The raw carry-over weights are:

$$
w_l = \alpha^{\,l}, \qquad l = 0, 1, 2, \dots
$$

So a single burst of spend leaves `1` in the same week, `alpha` next week,
`alpha^2` the week after, and so on:

![Adstock decay](img/adstock.png){ width="640" }

A large `alpha` keeps working for many weeks (think TV); a small `alpha` fades
almost immediately (think search).

In practice the **normalized** version is used, which divides by the sum of the
weights:

$$
\text{adstock}(x)_t = \frac{\sum_{l=0}^{L-1} \alpha^{\,l}\, x_{t-l}}{\sum_{l=0}^{L-1} \alpha^{\,l}}
$$

The only difference is scaling: normalizing keeps a steady, unchanging spend at
the same level instead of inflating it. The plot above drops the denominator so
all channels start at 1 and the shape is easy to compare. The implementation is
[`bmmm.data.transforms.geometric_adstock`][bmmm.data.transforms.geometric_adstock].

## Saturation (diminishing returns)

Doubling spend does not double sales. Logistic saturation bends the response so it
flattens out as spend grows, controlled by a steepness `lambda`:

$$
\text{sat}(x) = \frac{1 - e^{-\lambda x}}{1 + e^{-\lambda x}}
$$

![Saturation curves](img/saturation.png){ width="640" }

See [`bmmm.data.transforms.logistic_saturation`][bmmm.data.transforms.logistic_saturation].

## How sales are built up

Each channel's contribution combines the two effects: take the spend, apply
carry-over, apply diminishing returns, then scale by a per-channel weight `beta`:

$$
\text{contribution}_{c,t} = \beta_c \, \text{sat}\big(\text{adstock}(x_{c,t})\big)
$$

Everything not driven by advertising is the **baseline**:

$$
\text{baseline}_t = \text{intercept} + \text{trend}_t + \text{seasonality}_t + \text{price effect}_t
$$

- **intercept**: constant organic sales (brand, loyal customers, distribution).
- **trend**: a slow drift up or down over time.
- **seasonality**: a repeating yearly pattern, often built from sine and cosine
  (Fourier) terms.
- **price effect**: how sales move with price. Price is a control variable, not a
  channel. Higher price usually lowers sales, so its coefficient is negative.
  Including it stops the model from mistaking a price change for an ad effect.

The trend and the price effect enter as **control variables**: extra regressors
that explain part of sales but are not advertising. This keeps their influence
from being wrongly credited to a channel.

Total sales are the baseline plus the channel contributions plus noise. Once the
model is fitted, it can split observed sales back into these parts, which is the
basis for measuring each channel and for planning budget.

## Bayesian inference in short

So far this is just a model structure. Fitting it the Bayesian way is what gives
every number an uncertainty range.

A classic machine-learning model returns one best value for each parameter. A
Bayesian model returns a whole range of plausible values. That is often what you
want in marketing: it is more useful to know a channel's ROAS is "around 2, give
or take 0.3" than just "2".

It works with three pieces:

- **Prior**: what we believe about a parameter before seeing the data. For
  example, an adstock retention is between 0 and 1, and an ad effect is positive.
- **Likelihood**: how well a given set of parameters explains the observed data.
- **Posterior**: the updated belief after combining the prior with the data. This
  is the answer, and it is a distribution, not a single number.

In one line: the posterior is the prior updated by the data. Two practical
benefits over a single point estimate:

- **Honest uncertainty.** Every output (a contribution, a ROAS, a recommended
  budget) comes with a range, so a confident result can be told apart from a
  shaky one.
- **A natural place for domain knowledge.** Facts we already know, such as "more
  ad spend does not reduce sales", go in directly as priors instead of being
  bolted on afterward.

## Priors

A good default is a **weakly informative** prior: strong enough to rule out
nonsense (a negative ad effect, a retention above 1), weak enough to let the data
decide the actual value. The shape of the prior is usually chosen to match the
valid range of the parameter:

- a value between 0 and 1 fits a **Beta** distribution,
- a value that must be positive fits a **Gamma** or **HalfNormal**,
- a value that can go either way fits a **Normal** centered at zero.

A simple way to sanity-check priors is the **prior predictive**: generate data
from the priors alone, before fitting, and confirm it lands in a plausible range
rather than, say, millions or negatives.

## Sampling: how the posterior is computed

The posterior usually cannot be written as a formula, so we draw samples from it
and work with those. The family of methods for this is **MCMC** (Markov chain
Monte Carlo): a chain walks through the space of parameter values and spends more
time where the posterior is high. The collected values form our picture of the
posterior.

Plain MCMC explores by trial and error, which is slow when there are many
parameters. **HMC** (Hamiltonian Monte Carlo) is smarter: it uses the gradient of
the posterior to take long, informed steps, like a ball rolling across the
landscape instead of hopping at random. **NUTS** (the No-U-Turn Sampler) is the
version of HMC that tunes itself: it works out how far to roll on each step and
stops when the path starts to double back, so there is no manual tuning.

NUTS is the standard choice for models that are fully continuous and
differentiable, which is the case here. Simpler samplers (random-walk MCMC, Gibbs)
would need many more draws to reach the same quality.

## Checking the sampler

Because sampling is approximate, we run several independent chains and check they
agree. The standard diagnostics:

- **R-hat** compares the variation between chains to the variation within each
  chain. Close to 1.0 means the chains reached the same answer. Above about 1.01
  is a warning.
- **Effective sample size (ESS)** is how many genuinely independent samples we
  have. Samples from one chain are correlated, so ESS is smaller than the raw
  count; a few hundred is usually enough for stable averages and intervals.
- **Divergences** are a NUTS-specific signal that the sampler struggled with the
  geometry of the posterior. Zero is what we want.

## Credible intervals

A **credible interval** is the Bayesian version of a confidence interval, but it
means what people usually expect: "there is a 94% probability the value is in this
range". The **highest density interval (HDI)** is the narrowest such range. The
94% level is just the default we use; it is arbitrary, the same way 95% is.

## The tools

A small stack does the work, each piece with a clear job:

- **PyMC** is the probabilistic programming library. You describe the model and
  its priors, and PyMC builds the math and runs the sampler.
- **PyMC-Marketing** is built on PyMC and adds the marketing-specific parts: the
  adstock and saturation transforms, the `MMM` model that wires them together with
  controls and seasonality, and helpers for contributions and ROAS.
- **NUTS**, run through **nutpie** (a fast implementation), draws the posterior
  samples.
- **ArviZ** is the diagnostics and plotting library: R-hat, ESS, and credible
  intervals come from it.
