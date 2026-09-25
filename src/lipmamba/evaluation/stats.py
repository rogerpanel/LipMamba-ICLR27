"""Statistical tests used in the manuscript: Friedman test over methods ×
seeds with Holm post-hoc (Demšar, JMLR 2006) and paired Wilcoxon with the
rank-biserial effect size r."""
from __future__ import annotations

import numpy as np
from scipy import stats


def friedman_holm(scores: dict[str, list[float]], reference: str | None = None) -> dict:
    """``scores[method] = [score_seed1, score_seed2, ...]`` (same seeds for all).

    Returns the Friedman statistic and p-value and Holm-adjusted pairwise
    Wilcoxon p-values of every method against ``reference`` (default: the
    method with the best mean)."""
    names = list(scores)
    mat = np.array([scores[n] for n in names])            # (methods, seeds)
    if mat.shape[1] < 2:
        raise ValueError("need ≥2 seeds")
    chi2, p = stats.friedmanchisquare(*mat)
    ref = reference or names[int(mat.mean(1).argmax())]
    ref_scores = np.array(scores[ref])
    pvals, effects, others = [], [], []
    for n in names:
        if n == ref:
            continue
        x = np.array(scores[n])
        d = ref_scores - x
        if np.allclose(d, 0):
            pw, r = 1.0, 0.0
        else:
            pw = stats.wilcoxon(ref_scores, x, zero_method="wilcox").pvalue if len(d) >= 3 else float("nan")
            n_pairs = len(d)
            ranks = stats.rankdata(np.abs(d[d != 0]))
            r_plus = ranks[d[d != 0] > 0].sum(); r_minus = ranks[d[d != 0] < 0].sum()
            r = (r_plus - r_minus) / (n_pairs * (n_pairs + 1) / 2)
        pvals.append(pw); effects.append(float(r)); others.append(n)
    # Holm step-down
    order = np.argsort(pvals)
    m = len(pvals)
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return {"friedman_chi2": float(chi2), "friedman_df": len(names) - 1, "friedman_p": float(p),
            "reference": ref,
            "pairwise": {o: {"wilcoxon_p": float(pv), "holm_p": float(a), "rank_biserial_r": e}
                         for o, pv, a, e in zip(others, pvals, adj, effects)}}


def mean_std(xs: list[float]) -> tuple[float, float]:
    a = np.asarray(xs, dtype=float)
    return float(a.mean()), float(a.std(ddof=1)) if a.size > 1 else 0.0
