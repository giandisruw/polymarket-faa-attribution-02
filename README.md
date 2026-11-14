# Polymarket FAA outage attribution — 4-hour-ahead quantile forecaster

Forecasts the Yes price of every market of Polymarket's FAA outage
attribution family at every minute of the 4 hours after each checkpoint,
as five quantiles (0.05, 0.25, 0.5, 0.75, 0.95), for the lowest mean
pinball loss. Competition: `polymarket-faa-attribution-02` on make.fun.

## Run

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts
```

Node 22, standard library only. Reads every `<id>.in` in the input
directory (`MAKEFUN_INPUT_DIR`, default `/input`) and writes `<id>.out`
to the output directory (`MAKEFUN_OUTPUT_DIR`, default `/output`) as
`{"forecasts":[{"id","q"}]}`, `q` exactly 240 rows of the five quantiles.
Deterministic, offline, no network, no model calls, no secrets — a case
runs in well under a second (limit 8 s).

## Method

Ported from the sibling competition `polymarket-nvidia-ceo-forecast-01`'s
v3 forecaster (same case schema `makefun/polymarket-case/v1`, same output
format; the practice cases here are that event's cases re-cut byte for
byte). The forecaster is generic over the case's markets — no market
count, name or slug is assumed.

1. **Median = last price.** At 4-hour horizons the price is a
   near-martingale: every smoothing, drift extrapolation and
   mean-reversion rule tested on the practice cases was strictly worse
   than the last tick (a 30-min EWMA was ~48% worse on mean |error|).
2. **Sum-to-1 renormalization, clamped.** Medians move halfway toward
   summing to 1, then are shrunk by 0.995 (a calibration fix: the
   earlier median's empirical coverage was 0.41 vs a 0.50 target).
   The clamp is the one deliberate change from v3: the NVIDIA markets
   are mutually exclusive, so their Yes prices should sum to 1, but the
   FAA final's 4 markets are **not** (Iran is a foreign actor; a hack
   could be by a foreign actor), so their prices need not sum to 1.
   The renorm factor is therefore clamped to [0.9, 1.1] — it keeps the
   full practice-proven benefit when the sum is near 1 and bounds the
   distortion to ±10% when it is not.
3. **Per-market adaptive spread.** Sigma is a recency-weighted (EWMA,
   half-life 60 min) RMS of the market's own 1-minute changes, ignoring
   jumps (|d| > 0.02), × 0.8, floored at 0.0003/min. Recent volatility
   predicts the next 4 h better than a flat 240-minute window.
4. **Horizon growth.** `spread(h) = sqrt((0.8·sigma·sqrt(h))² + (2e-5·h)²)`
   — random-walk growth plus a linear jump-risk term.
5. **Shape.** Normal quantiles, interquartile z-values narrowed 0.7×
   and the 5% (downside) tail widened 1.3×: calibration showed the
   middle band too wide (realized IQR held 65% of outcomes vs a 50%
   target) while the 90% interval was about right (88% vs 90%).

Everything is clamped to [0,1], rounded to 6 decimals, and forced
non-decreasing per row.

## Practice evidence (local scoring, p00–p04)

The kit ships no answers and `grade-local` refuses this competition
(the scorer and reference data stay on the server), so the practice was
scored locally exactly as the sealed scorer does — mean pinball loss
×100 over every (case, market, minute, tau) — with ground truth taken
from the next case's price history (each case's checkpoint is exactly
240 minutes after the previous one). p05 has no following case in the
kit, so it was checked for validity and runtime only.

| case | this entry | house baseline | median err | cov90 |
|------|-----------|----------------|-----------|-------|
| p00  | 0.4244    | 0.5361         | 1.2709    | 0.834 |
| p01  | 0.3076    | 0.3804         | 0.7781    | 0.919 |
| p02  | 0.3460    | 0.4084         | 1.0313    | 0.844 |
| p03  | 0.2570    | 0.3686         | 0.7190    | 0.973 |
| p04  | 0.2086    | 0.2988         | 0.5902    | 0.894 |
| **mean** | **0.3019** | **0.3925** | — | — |

Beats the house baseline on every case, ~23% better overall. Variants
tested on the same five cases: unclamped v3 port 0.3026, no renorm
0.3124, renorm gated to sums in [0.8, 1.25] 0.3026 (identical to v3 on
practice). The clamp was adopted as a safety bound for the final, not
as a tuned improvement — its −0.0007 gain over v3 is within noise.

**Honest caveats.** The practice is another event's instance (NVIDIA);
the rules state its score is no guide to the final. About 50 variants
were tested on the sibling build and rejected (sigma shrinkage to
pooled, MAD sigma, sigma caps, conditional drift, logit-normal
quantiles, jump-cluster spreads, vol-trend ratios, OU saturation,
Student-t tails, empirical quantiles — all flat or worse).

## Files

- `main.ts` — the forecaster (Node 22, stdlib only)
- `README.md` — this file
- `LICENSE` — MIT

## License

MIT — see `LICENSE`.
