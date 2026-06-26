#!/usr/bin/env python3
"""Statistical significance of the accuracy result — McNemar + Wilson CIs.

Reads results/accuracy.jsonl (178 paired outcomes: per question, was Korely's
selected block correct, and was the equal-budget recency window correct). Computes:

  - the 2x2 paired contingency table (concordant + discordant pairs)
  - McNemar's test, two-sided, two ways:
      * exact binomial  (the defensible one for small discordant counts)
      * chi-square with Yate's continuity correction (the textbook formula)
  - Wilson 95% score intervals for each system's accuracy
  - 95% CI for the paired accuracy difference

Pure stdlib (math only) so anyone can reproduce it from a clone without scipy:

  python3 scripts/stats.py
"""
from __future__ import annotations
import json
import math
import os
import sys

Z = 1.959963985  # standard normal quantile for a two-sided 95% interval


def wilson(x: int, n: int) -> tuple[float, float]:
    """95% Wilson score interval for a binomial proportion."""
    if n == 0:
        return (0.0, 0.0)
    p = x / n
    denom = 1 + Z * Z / n
    center = (p + Z * Z / (2 * n)) / denom
    half = (Z / denom) * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n))
    return (center - half, center + half)


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact binomial McNemar p-value.

    Under H0 the two systems are equally good, so each of the (b+c) discordant
    pairs is a fair coin: b ~ Binomial(b+c, 1/2). p = 2 * P(X <= min(b, c)).
    """
    m = b + c
    if m == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(m, i) for i in range(k + 1)) * (0.5 ** m)
    return min(1.0, 2 * tail)


def mcnemar_chi2(b: int, c: int) -> tuple[float, float]:
    """McNemar chi-square with continuity correction + its 1-df p-value."""
    if b + c == 0:
        return (0.0, 1.0)
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    p = math.erfc(math.sqrt(chi2 / 2))  # survival of chi-square, 1 df
    return (chi2, p)


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(here, "..", "results", "accuracy.jsonl")
    base = os.path.splitext(os.path.basename(path))[0]
    out_name = "stats.json" if base == "accuracy" else f"stats_{base.replace('accuracy_', '')}.json"
    rows = [json.loads(line) for line in open(path) if line.strip()]

    a = b = c = d = 0  # both right / korely-only / window-only / both wrong
    for r in rows:
        k, w = bool(r["korely_correct"]), bool(r["window_correct"])
        if k and w:
            a += 1
        elif k and not w:
            b += 1
        elif w and not k:
            c += 1
        else:
            d += 1
    n = a + b + c + d

    korely_x, window_x = a + b, a + c
    korely_acc, window_acc = korely_x / n, window_x / n
    diff = korely_acc - window_acc

    # 95% CI for the paired difference of proportions (Agresti).
    var = ((b + c) - (b - c) ** 2 / n) / (n * n)
    se = math.sqrt(var)
    diff_lo, diff_hi = diff - Z * se, diff + Z * se

    p_exact = mcnemar_exact(b, c)
    chi2, p_chi2 = mcnemar_chi2(b, c)
    k_acc = wilson(korely_x, n)
    w_acc = wilson(window_x, n)

    out = {
        "n": n,
        "contingency": {
            "both_correct": a,
            "korely_only_correct": b,
            "window_only_correct": c,
            "both_wrong": d,
        },
        "korely_accuracy": round(100 * korely_acc, 1),
        "korely_accuracy_ci95": [round(100 * k_acc[0], 1), round(100 * k_acc[1], 1)],
        "window_accuracy": round(100 * window_acc, 1),
        "window_accuracy_ci95": [round(100 * w_acc[0], 1), round(100 * w_acc[1], 1)],
        "difference_pts": round(100 * diff, 1),
        "difference_ci95_pts": [round(100 * diff_lo, 1), round(100 * diff_hi, 1)],
        "mcnemar_b_korely_only": b,
        "mcnemar_c_window_only": c,
        "mcnemar_exact_p": p_exact,
        "mcnemar_chi2": round(chi2, 1),
        "mcnemar_chi2_p": p_chi2,
    }

    os.makedirs(os.path.join(here, "..", "results"), exist_ok=True)
    with open(os.path.join(here, "..", "results", out_name), "w") as fh:
        json.dump(out, fh, indent=2)

    def fmt_p(p: float) -> str:
        return f"{p:.2e}" if p < 1e-4 else f"{p:.4f}"

    print(f"n = {n} paired questions (LongMemEval oracle)\n")
    print("                       window WRONG   window RIGHT")
    print(f"  korely RIGHT              b = {b:<3}        a = {a}")
    print(f"  korely WRONG             d = {d:<3}        c = {c}")
    print()
    print(f"  Korely accuracy    {100*korely_acc:5.1f}%   95% CI [{100*k_acc[0]:.1f}, {100*k_acc[1]:.1f}]")
    print(f"  Window accuracy    {100*window_acc:5.1f}%   95% CI [{100*w_acc[0]:.1f}, {100*w_acc[1]:.1f}]")
    print(f"  Difference        +{100*diff:5.1f} pts 95% CI [{100*diff_lo:.1f}, {100*diff_hi:.1f}]")
    print()
    print(f"  McNemar discordant pairs: b={b} (korely only), c={c} (window only)")
    print(f"  McNemar exact binomial      p = {fmt_p(p_exact)}")
    print(f"  McNemar chi2 (cont. corr.)  chi2 = {chi2:.1f},  p = {fmt_p(p_chi2)}")
    print(f"\nWrote results/{out_name}")


if __name__ == "__main__":
    main()
