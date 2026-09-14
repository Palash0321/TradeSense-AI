# TradeSense-AI Cross-Index Research Matrix

Generated: `2026-09-14 22:01:43`

## Research Scope

This document consolidates previously validated independent index research. It is a research artifact only.

**No strategy is combined across indexes. No transferability is assumed. No production strategy is modified.**

## Baseline Comparison

| Index | Trades | Wins | Losses | Win Rate | PF | Total R | Status |
|---|---:|---:|---:|---:|---:|---:|---|
| NIFTY 50 | 22 | 6 | 16 | 27.27% | 0.674 | -5.50R | RESEARCH ONLY |
| SENSEX | 21 | 6 | 15 | 28.57% | 0.811 | -2.75R | RESEARCH ONLY |
| BANK NIFTY | 24 | 5 | 19 | 20.83% | 0.518 | -9.20R | RESEARCH ONLY |
| NIFTY MIDCAP SELECT | 13 | 4 | 9 | 30.77% | 0.755 | -2.25R | RESEARCH ONLY |

## Independent Index Findings

### NIFTY 50

- **Data source:** ^NSEI
- **Data status:** PASS
- **Research period:** 2021-01-01 → 2026-09-11
- **Rows:** 1408
- **Baseline:** 22 trades, 27.27% win rate, PF 0.674, -5.50R
- **Directional finding:** LONG +2.01R / PF 1.214; SHORT -7.51R. Short-side weakness is visible but sample is small.
- **Setup finding:** LC +2.27R, LR -0.26R, SC -2.17R, SR -5.34R. SR is the weakest setup in this sample.
- **Development finding:** Early-adverse 5-bar / 0.25R filtering was diagnostic only. 15 triggered trades -4.27R versus 7 non-triggered trades -1.23R.
- **Counterfactual finding:** LC exit-path counterfactual improved baseline by +2.70R, but robustness was not established.
- **Robustness:** ROBUSTNESS_NOT_ESTABLISHED
- **OOS status:** WAITING_FOR_GENUINE_OOS_TRADES
- **Candidate hypothesis:** Early-adverse behavior and LC exit-path management remain research candidates only.
- **Deployment blocker:** Insufficient index-level sample and no genuine forward validation.
- **Verdict:** **RESEARCH ONLY**

### SENSEX

- **Data source:** ^BSESN
- **Data status:** PASS
- **Research period:** 2021-01-01 → 2026-09-11
- **Rows:** 1405
- **Baseline:** 21 trades, 28.57% win rate, PF 0.811, -2.75R
- **Directional finding:** LONG +3.63R / PF 1.586; SHORT -6.38R. Short-side weakness is substantial in-sample but sample is small.
- **Setup finding:** LC +2.70R, LR +0.93R, SC +0.80R, SR -7.18R. SR is the dominant negative setup.
- **Development finding:** Early-adverse 5-bar / 0.25R diagnostic produced only +0.49R improvement and is therefore weak.
- **Counterfactual finding:** SR exit counterfactual improved by +2.02R, but depended on a very small trigger sample and failed leave-one-out robustness.
- **Robustness:** INSUFFICIENT_SAMPLE
- **OOS status:** NO_DEPLOYMENT_VALIDATION
- **Candidate hypothesis:** SR development/exit behavior may justify future research, but no implementation is supported.
- **Deployment blocker:** Tiny setup sample and insufficient robustness/OOS evidence.
- **Verdict:** **RESEARCH ONLY**

### BANK NIFTY

- **Data source:** ^NSEBANK
- **Data status:** PASS
- **Research period:** 2021-01-01 → 2026-09-11
- **Rows:** 1407
- **Baseline:** 24 trades, 20.83% win rate, PF 0.518, -9.20R
- **Directional finding:** LONG 11 trades versus SHORT 13 trades; overall short-side and continuation weakness are notable.
- **Setup finding:** LC 9 trades, SC 8, SR 5, LR 2. SC produced 0W / -8.19R; SR produced 1W / 4L / -2.13R.
- **Development finding:** All 8 SC trades triggered the 5-bar / 0.25R early-adverse condition and all were losses.
- **Counterfactual finding:** SC early-adverse filtering showed improvement across all 24 tested parameter cells, all 6 years, and all 8 leave-one-out tests; however, only 8 SC trades exist.
- **Robustness:** STRONG_IN_SAMPLE_FAILURE_SIGNAL_BUT_TINY_SAMPLE
- **OOS status:** WAITING_FOR_GENUINE_OOS_TRADES
- **Candidate hypothesis:** SC + early-adverse behavior is a strong candidate for frozen forward validation only.
- **Deployment blocker:** Only 8 SC trades and zero genuine post-holdout trades so far.
- **Verdict:** **RESEARCH ONLY**

### NIFTY MIDCAP SELECT

- **Data source:** Nifty Indices official historical endpoint
- **Data status:** PASS
- **Research period:** 2022-01-10 → 2026-09-11
- **Rows:** 1160
- **Baseline:** 13 trades, 30.77% win rate, PF 0.755, -2.25R
- **Directional finding:** LONG 8 trades / -0.14R; SHORT 5 trades / -2.11R. Sample is insufficient for directional conclusions.
- **Setup finding:** LC +0.89R, LR -1.03R, SC -2.05R, SR -0.06R. No setup has enough trades for deployment inference.
- **Development finding:** Clear favorable-development gradient: trades reaching 1.25R, 1.50R, and 1.75R had progressively stronger outcomes. Threshold arrival can be very late.
- **Counterfactual finding:** Exit counterfactual improved baseline by +2.11R. Sensitivity showed a threshold gradient, but only 13 trades were available.
- **Robustness:** INSUFFICIENT_SAMPLE
- **OOS status:** WAITING_FOR_GENUINE_OOS_TRADES
- **Candidate hypothesis:** Favorable-development thresholds may be useful as a future development-state feature, not as an immediate exit rule.
- **Deployment blocker:** Only 13 historical trades and no genuine forward trades.
- **Verdict:** **RESEARCH ONLY**

## Cross-Index Findings

### Baseline strategy quality

**Finding:** The current baseline is negative on all four independently tested indexes.

**Interpretation:** The baseline strategy is not yet suitable for deployment as a universal index strategy.

**Status:** `BLOCKED`

### Data integrity

**Finding:** NIFTY 50, SENSEX, and BANK NIFTY official/Yahoo index datasets passed integrity checks. MIDSELECT official Nifty Indices data was validated from 2022-01-10 onward.

**Interpretation:** No current data-integrity issue blocks index research.

**Status:** `PASS`

### Directional behavior

**Finding:** Long-side results are generally stronger than short-side results in these small index samples.

**Interpretation:** Potential recurring behavior, but not enough evidence to create a universal long-only or short-filter rule.

**Status:** `RESEARCH ONLY`

### Setup behavior

**Finding:** Different indexes exhibit different dominant weak setups.

**Interpretation:** Setup-specific behavior appears index-dependent. Do not transfer exclusions between indexes.

**Status:** `INDEX-SPECIFIC`

### Early-adverse behavior

**Finding:** Early-adverse behavior is strongly interesting in some index/setup combinations, especially BANK NIFTY SC and the previously validated stock research.

**Interpretation:** Promising failure-development feature, but must be frozen and independently validated per index.

**Status:** `OOS VALIDATION REQUIRED`

### Favorable development

**Finding:** MIDSELECT shows a strong relationship between favorable development and eventual trade outcome.

**Interpretation:** Useful as a research feature; not evidence for changing exits.

**Status:** `RESEARCH ONLY`

### Exit counterfactuals

**Finding:** Counterfactual exit improvements appear on multiple indexes, but robustness varies and samples are small.

**Interpretation:** Counterfactual improvement alone is insufficient for implementation.

**Status:** `NO IMPLEMENTATION`

### OOS evidence

**Finding:** Current forward ledgers contain no genuine post-start completed trades for the newly established BANK NIFTY and MIDSELECT holdouts.

**Interpretation:** There is currently no legitimate unseen-trade evidence on which to accept or reject those frozen hypotheses.

**Status:** `WAITING`

### Transferability

**Finding:** No hypothesis has yet demonstrated sufficient cross-index generalization to justify automatic transfer.

**Interpretation:** Every hypothesis must remain index-specific until independently validated.

**Status:** `BLOCKED`

### Production strategy

**Finding:** No research result currently meets the standard required to modify the production strategy.

**Interpretation:** Production remains unchanged.

**Status:** `LOCKED`

## Hypothesis Classification

| Hypothesis / Observation | Classification | Action |
|---|---|---|
| NIFTY 50 early-adverse behavior | Promising research candidate | Require genuine OOS validation |
| NIFTY 50 LC exit-path behavior | Interesting but robustness incomplete | Do not implement |
| SENSEX SR weakness | Index-specific / insufficient sample | No implementation |
| BANK NIFTY SC + early adverse | Strong in-sample failure signal | Frozen forward validation |
| MIDSELECT favorable-development gradient | Interesting development feature | Do not use as exit rule yet |
| Cross-index strategy transfer | Not established | Blocked |

## Current OOS Gate

The next meaningful evidence must come from genuine unseen post-holdout trades. Existing historical trades must not be reclassified as OOS trades.

Current forward collectors:

- BANK NIFTY: waiting for genuine post-2026-09-14 completed trades.
- MIDSELECT: waiting for genuine post-2026-09-14 completed trades.

No tuning should be performed using those future observations.

## Production Decision

**NO PRODUCTION STRATEGY CHANGE.**

The current evidence supports continued research and frozen forward validation, not deployment.

Each index remains an independent validation problem.

## Final Research Verdict

**TradeSense-AI is not yet ready to deploy an index-level strategy based on these findings.**

The research has successfully identified several potentially valuable behaviors, but none currently satisfies the complete chain of evidence:

baseline → hypothesis → frozen rule → robustness → genuine OOS validation → independent confirmation → deployment

The project should now prioritize genuine forward evidence rather than further mining of the same historical samples.
