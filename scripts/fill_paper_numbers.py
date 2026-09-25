#!/usr/bin/env python
"""Render the LaTeX replacement text for the manuscript from the JSON outputs
of the experiment scripts.  Nothing is invented: a snippet is only rendered
when its JSON exists, otherwise the item is listed as still open.

Inputs (all optional, under --runs):
  todo1_ell_star.json, todo2_adaptive.json, todo5_fig2.json, todo4_baselines.json,
  lipmamba-130m_init.provenance.json, perplexity_overhead.json,
  review/*.json   (scripts/run_review_experiments.py, one file per seed / model)

    python scripts/fill_paper_numbers.py --runs runs --out paper/todo_snippets.tex
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path


def load(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def mean(xs):
    xs = [x for x in xs if x is not None]
    return st.mean(xs) if xs else float("nan")


def review_snippets(files: list[Path]) -> str:
    """Aggregate run_review_experiments.py outputs (mean over seeds) into LaTeX."""
    runs = [json.loads(f.read_text()) for f in files]
    if not runs:
        return ""
    seeds = sorted({r["provenance"]["seed"] for r in runs})
    lengths = [row["ell"] for row in runs[0]["pacc_vs_length"]]
    out = [f"%% ---- run_review_experiments.py: {len(runs)} run(s), seeds {seeds} ----\n"]

    # adaptive-attack table (PACC, %), rows = ℓ, cols = attacks
    cols = ["z_hispa", "m_hispa", "adaptive_saturate", "adaptive_overwrite", "adaptive_margin"]
    out.append(r"""\begin{table}[t]\centering
\caption{Poisoned accuracy (PACC, \%) on RoBench-25 versus trigger length under the published HiSPA triggers and the three adaptive white-box objectives that optimise directly against the clamp (mean over seeds). Worst case is the minimum over the five columns.}
\label{tab:adaptive}\small\setlength{\tabcolsep}{4pt}
\begin{tabular}{@{}lcccccc@{}}\toprule
$\ell$ & Z-HiSPA & M-HiSPA & adaptive: saturate & adaptive: overwrite & adaptive: margin & worst case \\ \midrule
""")
    for L in lengths:
        vals = []
        for c in cols:
            vals.append(100 * mean([next(r["pacc"] for r in run["pacc_vs_length"] if r["ell"] == L)
                                    if False else next(row[c]["pacc"] for row in run["pacc_vs_length"] if row["ell"] == L) for run in runs]))
        wc = 100 * mean([next(row["worst_case_pacc"] for row in run["pacc_vs_length"] if row["ell"] == L) for run in runs])
        out.append(f"{L} & " + " & ".join(f"{v:.1f}" for v in vals) + f" & \\textbf{{{wc:.1f}}} \\\\\n")
    out.append("\\bottomrule\\end{tabular}\\end{table}\n")

    # adaptive paragraph at ℓ = 24
    r24 = [next(row for row in run["pacc_vs_length"] if row["ell"] == 24) for run in runs if any(row["ell"] == 24 for row in run["pacc_vs_length"])]
    if r24:
        sat = mean([r["adaptive_saturate"]["delta_saturation_fraction"] for r in r24]) * 100
        ret = mean([r["adaptive_saturate"]["retention_ratio_mean"] for r in r24])
        ovw = mean([r["adaptive_overwrite"]["overwrite_distance_mean"] for r in r24])
        pm = mean([r["adaptive_margin"]["pacc"] for r in r24]) * 100
        pw = mean([r["worst_case_pacc"] for r in r24]) * 100
        pz = mean([r["m_hispa"]["pacc"] for r in r24]) * 100
        out.append(r"""%% Adaptive-attack paragraph (Experiments / Discussion)
\paragraph{Adaptive attack.} Following \citet{athalye2018obfuscated,carlini2019evaluating}, a white-box adversary with gradient access through \eqref{eq:clamp} optimises the trigger directly against the defence: (i) a GCG-style objective that maximises the mean step over the trigger and minimises the post-trigger state norm, (ii) a content-overwrite objective that maximises $\norm{\h-\h^{\mathrm{clean}}}$ at fixed norm, and (iii) a margin objective on the verbalised logits (\cref{tab:adaptive}). At $\ell=24$ the saturating adversary pins $\Delta_t$ to $%(sat).0f\%%$ of $\Delta_{\max}$ and reduces the retention ratio to $%(ret).2f$, the content adversary moves the state by $%(ovw).2f$ relative units while keeping its norm, and the margin adversary lowers PACC from $%(pz).1f\%%$ (M-HiSPA) to $%(pm).1f\%%$; the worst case over all attacks is $%(pw).1f\%%$, which is the number to read as the model's robustness at this length.
""" % dict(sat=sat, ret=ret, ovw=ovw, pz=pz, pm=pm, pw=pw))

    # Table-1 rows: LipMamba (this run), GloRo-Mamba, randomized smoothing
    acc = 100 * mean([r["clean"]["acc"] for r in runs]); ece = mean([r["clean"]["ece"] for r in runs])
    ll = 100 * mean([r["ll_acc"]["0.18"] for r in runs]); gc = 100 * mean([r["global_cert_acc"]["0.18"] for r in runs])
    pw24 = 100 * mean([next(row["worst_case_pacc"] for row in run["pacc_vs_length"] if row["ell"] == 24) for run in runs]) if r24 else float("nan")
    out.append("%% Table 1 rows (mean over seeds). Columns: ACC & PACC(worst-case, ℓ=24) & ASR & LL-Acc@0.18 & ECE & Lat.\n")
    out.append(f"\\modelname{{}} (this run) & {acc:.1f} & {pw24:.1f} & ASR & {ll:.1f} & {ece:.3f} & lat \\\\   %% global-constant certified acc@0.18 = {gc:.1f}\n")
    if all("gloro_mamba" in r for r in runs):
        g = runs
        out.append("GloRo-Mamba (head only) & %.1f & %.1f & ASR & %.1f & %.3f & lat \\\\\n" % (
            100 * mean([r["gloro_mamba"]["acc"] for r in g]), 100 * mean([r["gloro_mamba"]["pacc_mhispa_24"] for r in g]),
            100 * mean([r["gloro_mamba"]["ll_acc_0.18"] for r in g]), mean([r["gloro_mamba"]["ece"] for r in g])))
    if all("randomized_smoothing" in r for r in runs):
        rs = 100 * mean([r["randomized_smoothing"]["certified_acc"]["0.18"] for r in runs])
        sig = runs[0]["randomized_smoothing"]["sigma"]
        out.append(f"Randomized smoothing ($\\sigma={sig}$) on \\modelname{{}} & {acc:.1f} & --- & --- & {rs:.1f} (certified) & --- & --- \\\\\n")
        out.append("%% RS certified-accuracy curve: " + " ".join(f"({k},{100*mean([r['randomized_smoothing']['certified_acc'][k] for r in runs]):.1f})" for k in runs[0]["randomized_smoothing"]["certified_acc"]) + "\n")
    out.append("%% LL-Acc curve: " + " ".join(f"({k},{100*mean([r['ll_acc'][k] for r in runs]):.1f})" for k in runs[0]["ll_acc"]) + "\n")
    return "".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs"); ap.add_argument("--out", default="paper/todo_snippets.tex")
    a = ap.parse_args()
    R = Path(a.runs)
    t1 = load(R / "todo1_ell_star.json"); t2 = load(R / "todo2_adaptive.json"); t3 = load(R / "lipmamba-130m_init.provenance.json")
    t4 = load(R / "todo4_baselines.json"); t5 = load(R / "todo5_fig2.json"); ppl = load(R / "perplexity_overhead.json")
    out, open_ = [], []
    out.append("% ---- Generated by scripts/fill_paper_numbers.py — paste over the corresponding text ----\n")

    if t1:
        w = t1["per_input_worst_case"]; d = t1["per_input_data_dependent"]; c = t1["constants"]
        out.append(r"""%% Remark 5 / Fig. 4 (ell*)
With the trained constraint values $(s_B,\Delta_{\max},\lambda_{\max},X_{\max})=(%(s_b)g,%(delta_max)g,%(lambda_max)g,%(x_max)g)$, $\rho_{\min}=%(rho_min).3f$ and $c=%(c).3g$, the worst-case bound \eqref{eq:lstar} certifies $\ell^\star\in[%(wmin).2f,%(wmax).2f]$ (median $%(wmed).2f$) over the observed pre-trigger norms (median $%(h0).2f$) at $\alpha_{\min}=%(alpha)g$; the data-dependent refinement gives $\ell^\star$ with $5$th percentile $%(dp05).2f$ and median $%(dmed).2f$. Certifying $\ell^\star\ge%(target)d$ would require $\Delta_{\max}\lambda_{\max}\le%(req).3f$ (currently $%(cur).3f$).
""" % dict(s_b=c["s_b"], delta_max=c["delta_max"], lambda_max=c["lambda_max"], x_max=c["x_max"], rho_min=c["rho_min"], c=c["c"],
           wmin=w["ell_star_min"], wmax=w["ell_star_max"], wmed=w["ell_star_median"], h0=w["h0_norm_median"], alpha=t1["alpha_min"],
           dp05=d["p05"] or float("nan"), dmed=d["median"] or float("nan"),
           target=t1["required_delta_max_times_lambda_max_for_target"]["target_ell"], req=t1["required_delta_max_times_lambda_max_for_target"]["product"],
           cur=t1["required_delta_max_times_lambda_max_for_target"]["current_product"]))
    else:
        open_.append("ell* distribution: run scripts/todo1_ell_star.py")

    review_files = sorted((R / "review").glob("*.json")) if (R / "review").exists() else []
    if review_files:
        out.append(review_snippets(review_files))
    elif t2:
        r24 = next((r for r in t2["rows"] if r["trigger_length"] == 24), t2["rows"][-1])
        s = r24["adaptive_saturate_continuous"]; o = r24["adaptive_overwrite_continuous"]; m = r24["adaptive_margin_continuous"]
        out.append(r"""%% Adaptive attack (from todo2_adaptive_attack.py; prefer run_review_experiments.py)
At $\ell=%(L)d$ the saturating adversary pins $\Delta_t$ to $%(sat).0f\%%$ of $\Delta_{\max}$ and reduces the retention ratio to $%(ret).2f$, the content adversary moves the state by $%(ovw).2f$ relative units while keeping its norm, and the margin adversary flips $%(flip).0f\%%$ of predictions.
""" % dict(L=r24["trigger_length"], sat=100 * s["delta_saturation_fraction"], ret=s["retention_ratio_mean"], ovw=o["overwrite_distance_mean"], flip=100 * m["label_flip_rate"]))
    else:
        open_.append("adaptive attack / GloRo row / randomized smoothing: run scripts/run_review_experiments.py per seed")

    if t3:
        out.append(r"""%% Implementation (base checkpoints)
\modelname{}-130M and -370M are initialised from \texttt{%(base)s} (revision \texttt{%(rev)s}); a fraction $%(clip).2f$ of the eigenvalues $|\lambda_i|$ fell outside $[\lambda_{\min},\lambda_{\max}]$ and were clipped at conversion.
""" % dict(base=t3["base_model_id"], rev=t3["base_revision"], clip=t3["conversion_stats"]["A_clipped_fraction_mean"]))
        if ppl:
            out.append("%% measured overhead: base PPL %.2f, constrained PPL %.2f, overhead %.1f%%\n" % (ppl["ppl_base"], ppl["ppl_constrained"], ppl["overhead_percent"]))
    else:
        open_.append("base-checkpoint provenance: run scripts/todo3_init_from_hf_mamba.py and scripts/perplexity_overhead.py")

    if t4 and not review_files:
        def row(name, label):
            t = t4["table_mean_std"][name]
            return f"{label} & {100*t['ACC'][0]:.1f} & {100*t['PACC'][0]:.1f} & --- & {100*t['LL_Acc_0.18'][0]:.1f} & {t['ECE'][0]:.3f} & {t['latency_ms_per_token'][0]:.2f}\\,ms \\\\"
        out.append("%% Table 1 rows from todo4 (mean over seeds %s)\n" % t4["seeds"])
        for n, l in (("unconstrained_mamba", "Mamba (unconstrained)"), ("gloro_mamba", "GloRo-Mamba (head only)"), ("lipmamba_arch", "\\modelname{}")):
            out.append(row(n, l) + "\n")

    if t5:
        r = t5["rows"][-1]; pg = t5["pgfplots"]
        out.append(r"""%% Fig. 2 coordinates (log10 y): worst %(pw)s | data-dep %(pd)s | op-norm %(po)s | empirical LB %(pe)s
%% at L=%(L)d: worst 10^%(w).0f, data-dependent 10^%(d).1f, op-norm 10^%(o).1f, empirical LB 10^%(e).1f
""" % dict(L=r["depth"], w=r["log10_worst_case"], d=r["log10_data_dependent_max"], o=r["log10_op_norm_product"], e=r["log10_empirical_lb_max"],
           pw=pg["worst_case"], pd=pg["data_dependent"], po=pg["op_norm"], pe=pg["empirical_lb"]))
    else:
        open_.append("Fig. 2: run scripts/todo5_fig2_lipschitz_depth.py")

    if open_:
        out.append("\n% STILL OPEN:\n" + "".join(f"%   - {o}\n" for o in open_))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text("".join(out))
    print("".join(out))


if __name__ == "__main__":
    main()
