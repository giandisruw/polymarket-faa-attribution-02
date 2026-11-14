// Polymarket FAA outage attribution 4-hour-ahead quantile forecaster.
//
// Ported from the sibling competition polymarket-nvidia-ceo-forecast-01's v3
// forecaster (same case schema makefun/polymarket-case/v1, same .out format;
// the practice cases here are that event's cases re-cut byte for byte).
// The final's universe is different — 4 FAA attribution markets, not 24
// mutually exclusive CEO candidates — so the one deliberate change from v3
// is in the sum-to-1 renormalization (see below).
//
// Model (own design, fitted only on the shape of the practice history, not on
// any single outcome):
//   * Median: the market's last price, moved halfway toward the sum-to-1
//     renormalization, then shrunk by 0.995. On this event shape the price
//     is a near-martingale at 4h horizons: every smoothing, drift
//     extrapolation and mean-reversion rule tested on the practice cases was
//     strictly worse than the last tick (EWMA-30 was ~48% worse on mean
//     |error|; conditional drift extrapolation was also worse on every
//     variant). The 0.995 shrink is a calibration fix: the v2 median's
//     empirical coverage was 0.41 against a 0.50 target (prices land below
//     the forecast median less often than half the time), i.e. the median
//     sat slightly high.
//   * Renormalization, CLAMPED (the change from v3): the NVIDIA markets are
//     mutually exclusive outcomes, so their Yes prices should sum to 1 —
//     and on the practice, half-renorm beats no renorm by 0.0098 (0.3026 vs
//     0.3124). The FAA final's 4 markets are NOT mutually exclusive (Iran
//     is a foreign actor; a hack could be by a foreign actor), so their
//     prices need not sum to 1 and an unbounded renorm could badly distort
//     the medians if the final's sum lands far from 1 (e.g. sum 0.6 would
//     push every median up 33%). The renorm factor is therefore clamped to
//     [0.9, 1.1]: it keeps the full practice-proven benefit when the sum is
//     near 1 (all six practice sums lie in 0.81-1.09, so the clamp only
//     binds on p02's 0.8075) and bounds the distortion to +-10% when it is
//     not. On the practice this scores 0.3019 vs v3's 0.3026 — the gain is
//     within noise, so the clamp is adopted as a safety bound for the
//     final, not as a tuned improvement.
//   * Spread: per-market, from the market's own recent 1-minute moves.
//     Recency-weighted (EWMA, half-life 60 min) RMS of 1-minute changes,
//     ignoring jumps (|d| > 0.02, which belong to the tail, not the core),
//     times 0.8, floored at 0.0003/min. Recent volatility predicts the next
//     4h better than a flat 240-minute window (EWMA-60 beat flat RMS on the
//     practice score; longer half-lives were worse).
//   * Growth: the spread widens like sqrt(h) (a random walk's), plus a small
//     term linear in h for jump risk, which accumulates linearly with the
//     horizon: spread(h) = sqrt((0.8*sigma*sqrt(h))^2 + (2e-5*h)^2).
//   * Shape: normal quantiles, except the interquartile z-values are
//     narrowed 0.7x and the 5% (downside) tail is widened 1.3x. Calibration
//     diagnostics on v2 showed the middle band too wide — the realized
//     interquartile range held 65% of outcomes against a 50% target, while
//     the 90% interval was about right (88% vs 90%) — so the shape is more
//     peaked than normal: narrower IQR, same tails.
//   * Everything is clamped to [0,1], rounded to 6 decimals, and made
//     non-decreasing; the program is deterministic (no randomness, no clock),
//     stdlib only, and runs a case in well under a second.
//
// Evidence (practice p00-p04, mean pinball loss x100; p05 has no following
// case in the kit, so no local ground truth — validity/runtime only):
//   house baseline (starter): 0.3925 (per case 0.5361 / 0.3804 / 0.4084 /
//     0.3686 / 0.2988). This entry: 0.3019 (per case 0.4244 / 0.3076 /
//     0.3460 / 0.2570 / 0.2086) — beats the house on every case, ~23%
//     better overall. The sealed house figure for all six practice cases is
//     0.363589; our p00-p04 mean is not directly comparable (different case
//     mix), but the same-case comparison above is.
//   HONEST CAVEATS: the practice is another event's instance (NVIDIA), and
//   the rules state its score is no guide to the final. The clamp's gain
//   over the unclamped v3 port (-0.0007) is within noise. About 50 variants
//   were tested on the sibling build and rejected (sigma shrinkage to
//   pooled, MAD sigma, sigma caps, conditional drift, logit-normal
//   quantiles, jump-cluster spreads, vol-trend ratios, OU saturation,
//   Student-t tails, empirical quantiles — all flat or worse).
//
// Input/output contract (program/v1): read every <id>.in in the input dir
// (MAKEFUN_INPUT_DIR, default /input), write <id>.out to the output dir
// (MAKEFUN_OUTPUT_DIR, default /output) as {"forecasts":[{"id","q"}]}, q
// exactly horizon_minutes rows of the five quantiles [0.05, 0.25, 0.5, 0.75,
// 0.95], one finite number in [0,1] per level, non-decreasing per row.
//
//   mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** The quantile levels the contest scores, in the order each row lists them. */
const QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.95];
/** Interquartile z-values narrowed 0.7x (v3: peaked shape, see header). */
const IQR_K = 0.7;
/** Standard normal quantiles at those levels; the 5% tail is widened 1.3x. */
const Z = [-1.644854 * 1.3, -0.67449 * IQR_K, 0, 0.67449 * IQR_K, 1.644854];

/** Half-life (minutes) of the EWMA weights in the per-market sigma. */
const SIGMA_HALF_LIFE_MIN = 60;
/** |1-min move| above this counts as a jump: excluded from the core sigma. */
const JUMP_CUT = 0.02;
/** Core-sigma multiplier (calibrated on practice: 0.8 beats 1.0 and 0.7). */
const SIGMA_MULT = 0.8;
/** Floor on the per-minute core sigma, for markets that barely ever move. */
const SIGMA_FLOOR = 0.0003;
/** Extra spread per minute of horizon, for jump risk (linear in h). */
const JUMP_PER_MIN = 0.00002;
/** How far the medians move toward summing to 1 (0 = no renorm, 1 = full). */
const RENORM = 0.5;
/** The renorm factor is clamped to this range: the final's 4 FAA markets are
 *  not mutually exclusive, so their prices need not sum to 1 — an unbounded
 *  renorm could badly distort the medians if the final's sum lands far from
 *  1. On the practice (sums 0.81-1.09) the clamp only binds on p02. */
const RENORM_FACTOR_MIN = 0.9;
const RENORM_FACTOR_MAX = 1.1;
/** Median shrink (v3: calibration fix for median coverage 0.41 vs 0.50). */
const MEDIAN_SHRINK = 0.995;

interface CaseMarket {
  id: string;
  prices: number[];
}

interface Case {
  horizon_minutes: number;
  quantiles: number[];
  markets: CaseMarket[];
}

/** x clamped to [0,1] and rounded to 6 decimals (0, never -0). */
const price6 = (x: number): number => Math.round(Math.min(1, Math.max(0, x)) * 1e6) / 1e6 + 0;

/**
 * A market's per-minute core sigma: recency-weighted (EWMA, half-life 60
 * min) RMS of its 1-minute changes, ignoring jumps. Falls back to the floor
 * when there is too little usable history.
 */
function coreSigma(prices: number[]): number {
  const n = prices.length;
  if (n < 2) return SIGMA_FLOOR;
  const dCount = n - 1;
  // EWMA decay: a = 1 - 0.5^(1/half-life); the most recent diff has weight 1.
  const a = 1 - Math.pow(0.5, 1 / SIGMA_HALF_LIFE_MIN);
  const decay = 1 - a;
  let sumW = 0;
  let sumWD2 = 0;
  for (let i = 0; i < dCount; i++) {
    const d = prices[i + 1] - prices[i];
    if (Math.abs(d) > JUMP_CUT) continue;
    // Weight of diff i: decay^(dCount - 1 - i); newest diff (i = dCount-1) = 1.
    const w = Math.pow(decay, dCount - 1 - i);
    sumW += w;
    sumWD2 += w * d * d;
  }
  if (sumW < 10) return SIGMA_FLOOR;
  return Math.max(SIGMA_FLOOR, SIGMA_MULT * Math.sqrt(sumWD2 / sumW));
}

/** The forecast rows for one market: horizon rows of the five quantiles. */
function forecast(median: number, sigma: number, horizon: number): number[][] {
  const rows: number[][] = [];
  for (let h = 1; h <= horizon; h++) {
    const spread = Math.sqrt((sigma * Math.sqrt(h)) ** 2 + (JUMP_PER_MIN * h) ** 2);
    const row = Z.map((z) => price6(median + z * spread));
    // Guarantee non-decreasing even after rounding/clamping.
    for (let j = 1; j < row.length; j++) {
      if (row[j] < row[j - 1]) row[j] = row[j - 1];
    }
    rows.push(row);
  }
  return rows;
}

mkdirSync(OUTPUT_DIR, { recursive: true });
// Every case file, in name order, so two runs write the same bytes.
const inputs = readdirSync(INPUT_DIR).filter((f) => f.endsWith('.in')).sort();
for (const file of inputs) {
  const id = file.slice(0, -'.in'.length);
  const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as Case;
  if (c.quantiles.length !== QUANTILES.length || c.quantiles.some((t, j) => t !== QUANTILES[j])) {
    throw new Error(`${file} asks for quantiles ${JSON.stringify(c.quantiles)}, not ${JSON.stringify(QUANTILES)}`);
  }
  // Medians: last price, moved halfway toward the sum-to-1 renormalization,
  // then shrunk by 0.995 (v3 calibration fix). The renorm factor is clamped
  // (see RENORM_FACTOR_MIN/MAX): the final's markets are not mutually
  // exclusive, so their sum need not be 1.
  const lasts = c.markets.map((m) => (m.prices.length > 0 ? m.prices[m.prices.length - 1] : 0.5));
  const sum = lasts.reduce((a, b) => a + b, 0);
  const rawFactor = sum > 0 ? 1 - RENORM + RENORM / sum : 1;
  const factor = Math.min(RENORM_FACTOR_MAX, Math.max(RENORM_FACTOR_MIN, rawFactor));
  const forecasts = c.markets.map((m, i) => ({
    id: m.id,
    q: forecast(lasts[i] * factor * MEDIAN_SHRINK, coreSigma(m.prices), c.horizon_minutes),
  }));
  writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify({ forecasts })}\n`);
}
