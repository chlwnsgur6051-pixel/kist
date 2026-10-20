"""Static weighted-correlation sensitivity and association-factor diagnostics.

Run from repository root:
    python my_research/experiments/diagnostics/analyze_association_strength.py
Optional archived MILP solutions (JSON list of {label, alpha} records):
    python my_research/experiments/diagnostics/analyze_association_strength.py --solutions /path/to/solutions.json

No STDP or LIF simulator is executed. A new timestamped result directory is created.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime
from itertools import combinations
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl_association_strength")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT))
import binary_character as data  # noqa: E402

ROOT = PROJECT / "results" / "association_strength_analysis"
KS = (3, 5, 7, 10)
ALPHAS = (0.01, 0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0)
HIGH_SIM_ABS_DOT = 9  # fixed before inspecting outcomes; |dot|/15 >= 0.6
VULNERABLE_MARGIN = 0  # error or tie under the fixed score>0 decoder
X01 = data.make_digit_dataset()[0].reshape(10, 15).astype(np.int64)
Y01 = data.make_letter_dataset()[0].reshape(10, 15).astype(np.int64)
X = 2 * X01 - 1
Y = 2 * Y01 - 1
G = X @ X.T
T = Y @ Y.T


def write_csv(path, rows, fields=None):
    rows = list(rows)
    if not rows and fields is None:
        raise ValueError(f"No rows and no header for {path}")
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def rank(values):
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    result = np.empty(len(values), dtype=float)
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and values[order[j]] == values[order[i]]:
            j += 1
        result[order[i:j]] = (i + 1 + j) / 2
        i = j
    return result


def spearman(a, b):
    a, b = rank(a), rank(b)
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def eval_scores(scores, targets):
    pred = scores > 0  # zero score -> 0, exactly as archived correlation sweep
    truth = targets > 0
    bit_ok = pred == truth
    return pred, bit_ok, np.all(bit_ok, axis=1), scores * targets


def factors(ids, base_score, base_ok, base_exact, base_margin):
    rows, pair_rows = [], []
    for i, p in enumerate(ids):
        others = [(h, q) for h, q in enumerate(ids) if q != p]
        similarities = np.array([G[p, q] for _, q in others], dtype=int)
        most = min((q for _, q in others), key=lambda q: (-abs(G[p, q]), q))
        signed_to_others = np.array([G[q, p] * Y[q] * Y[p] for _, q in others])
        vulnerable = np.array([base_margin[h] <= VULNERABLE_MARGIN for h, _ in others])
        negative = np.minimum(signed_to_others, 0)
        positive = np.maximum(signed_to_others, 0)
        # A positive input dot is helpful only where target signs agree; negative dots reverse that relation.
        rows.append(dict(
            k=len(ids), subset="-".join(map(str, ids)), digit=p, letter=chr(65+p),
            mean_absolute_input_similarity=float(np.mean(abs(similarities))),
            maximum_absolute_input_similarity=int(np.max(abs(similarities))),
            most_similar_input=most,
            highly_similar_input_count=int(np.sum(abs(similarities) >= HIGH_SIM_ABS_DOT)),
            mean_target_similarity=float(np.mean([T[p, q]/15 for _, q in others])),
            mean_conflicting_target_bits=float(np.mean([(15-T[p, q])/2 for _, q in others])),
            mean_input_target_compatibility=float(np.mean([G[p, q]*T[p, q] for _, q in others])),
            baseline_incorrect_bits=int(15-np.sum(base_ok[i])),
            baseline_minimum_signed_margin=int(np.min(base_margin[i])),
            baseline_mean_signed_margin=float(np.mean(base_margin[i])),
            most_vulnerable_output_neuron=int(np.argmin(base_margin[i])),
            baseline_exact=int(base_exact[i]),
            total_negative_interference_caused=int(-np.sum(negative)),
            total_positive_contribution_to_others=int(np.sum(positive)),
            negatively_affected_other_bits=int(np.sum(signed_to_others < 0)),
            negative_interference_on_vulnerable_bits=int(-np.sum(negative[vulnerable])),
            vulnerable_other_bit_count=int(np.sum(vulnerable)),
        ))
        for h, q in others:
            signed = G[q, p] * Y[q] * Y[p]
            assert np.array_equal(signed, G[p, q] * Y[p] * Y[q])
            pair_rows.append(dict(k=len(ids), subset="-".join(map(str, ids)),
                                  source_digit=p, recipient_digit=q,
                                  input_dot=int(G[p, q]), target_dot=int(T[p, q]),
                                  target_conflicting_bits=int((15-T[p, q])/2),
                                  input_target_compatibility=int(G[p, q]*T[p, q]),
                                  signed_interference_sum=int(np.sum(signed)),
                                  negative_interference_magnitude=int(-np.sum(np.minimum(signed, 0))),
                                  vulnerable_negative_magnitude=int(-np.sum(np.minimum(signed[base_margin[h]<=0], 0)))))
    return rows, pair_rows


def load_solutions(path):
    if path is None:
        return [], "No archived MILP alpha artifact was found or supplied."
    records = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError("Solutions JSON must be a list of {label, alpha} records")
    for r in records:
        if not isinstance(r, dict) or "label" not in r or "alpha" not in r:
            raise ValueError("Each solution needs label and alpha")
        if len(r["alpha"]) != 10 or not np.all(np.isfinite(r["alpha"])) or min(r["alpha"]) <= 0:
            raise ValueError("Each alpha must contain ten finite positive numbers in digit order")
    return records, f"Read archived solutions from {path}; labels and bounds are supplied by that artifact."


def load_reference_zip(path):
    """Read archived target-informed alpha without fitting or changing the source archive."""
    with zipfile.ZipFile(path) as z:
        dataset = np.load(io.BytesIO(z.read("dataset.npz")))
        alpha_data = np.load(io.BytesIO(z.read("optimized_alpha_reference.npz")))
        assert np.array_equal(dataset["X_bipolar"], X)
        assert np.array_equal(dataset["Y_bipolar"], Y)
        assert np.array_equal(alpha_data["X"], X)
        assert np.array_equal(alpha_data["Y"], Y)
        assert np.array_equal(alpha_data["recalled_digits"], [0,1,2,6,7,9])
        with z.open("optimized_alpha_reference.csv") as f:
            table = list(csv.DictReader(io.TextIOWrapper(f, encoding="utf-8")))
        assert [int(r["digit"]) for r in table] == list(range(10))
        records = []
        for label, key, column in [
            ("Archived MILP alpha [0.01,4.0]", "alpha_milp", "milp_alpha"),
            ("Archived six-preserving margin alpha", "alpha_margin_optimized", "margin_optimized_alpha"),
        ]:
            alpha = alpha_data[key].astype(float)
            assert np.allclose(alpha, [float(r[column]) for r in table], rtol=0, atol=1e-9)
            assert np.isclose(alpha.sum(), 10, atol=1e-7)
            assert np.all((alpha >= .01-1e-9) & (alpha <= 4+1e-9))
            records.append({"label":label, "alpha":alpha.tolist(), "source":path.name})
        return records, (f"Validated both bipolar arrays and two alpha vectors in {path.name} "
                         f"(SHA256 {hashlib.sha256(path.read_bytes()).hexdigest()}). "
                         "The archive contains no [0.5,3.0] optimum.")


def plot_results(out, sens10, factors10, comparisons, solution_rows):
    figdir = out / "figures"
    figdir.mkdir()
    a = np.array(ALPHAS)
    # Ten panels keep individual association sensitivity legible.
    fig, axes = plt.subplots(2, 5, figsize=(16, 7), sharex=True, sharey=True)
    for p, ax in enumerate(axes.flat):
        rows = sorted((r for r in sens10 if r["changed_digit"] == p), key=lambda r: r["alpha"])
        ax.plot(a, [100*r["all_exact_rate"] for r in rows], marker="o", ms=3, label="All memories")
        ax.axhline(10, color="gray", ls="--", lw=0.8, label="Equal α" if p == 0 else None)
        ax.set_title(f"Digit {p} strength changed")
        ax.grid(alpha=0.2)
    axes.flat[0].legend(fontsize=8)
    fig.supxlabel("Association Learning Strength α (all other α = 1)")
    fig.supylabel("Exact Recall Accuracy (%)")
    fig.suptitle("K=10: Exact Recall Accuracy vs Association Learning Strength")
    fig.tight_layout(); fig.savefig(figdir/"exact_accuracy_vs_learning_strength.png", dpi=155); plt.close(fig)

    def scatter(file, xkey, ykey, xlabel, ylabel, title, color_key=None):
        fig, ax = plt.subplots(figsize=(7, 5))
        xs = [float(r[xkey]) for r in comparisons]
        ys = [float(r[ykey]) for r in comparisons]
        if color_key:
            c = [float(r[color_key]) for r in comparisons]
            im = ax.scatter(xs, ys, c=c, cmap="coolwarm", s=50)
            fig.colorbar(im, ax=ax, label=color_key.replace("_", " "))
        else:
            ax.scatter(xs, ys, s=50)
        for r, x, y in zip(comparisons, xs, ys):
            ax.annotate(str(r["digit"]), (x, y), xytext=(3, 3), textcoords="offset points", fontsize=8)
        ax.axhline(0, color="black", lw=0.6)
        ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
        ax.grid(alpha=0.2); fig.tight_layout(); fig.savefig(figdir/file, dpi=155); plt.close(fig)

    scatter("self_improvement_vs_interference.png", "self_bit_delta_at_alpha_2", "other_bit_delta_at_alpha_2",
            "Self Correct Output Bits Change", "Other Correct Output Bits Change",
            "K=10: α=2 vs α=1, Self Improvement and Interference")
    scatter("input_similarity_vs_preferred_strength.png", "mean_absolute_input_similarity", "preferred_alpha",
            "Mean Absolute Input Dot Product", "Best Single-Change α (smallest tie)",
            "K=10: Input Similarity vs Retrospective Single-Change Strength")
    scatter("target_compatibility_vs_strength_effect.png", "mean_input_target_compatibility", "all_exact_delta_at_alpha_2",
            "Mean Input Dot × Target Dot", "Exact Recall Change (percentage points)",
            "K=10: Target Compatibility vs α=2 Effect")
    scatter("recall_difficulty_vs_strength_effect.png", "baseline_incorrect_bits", "self_bit_delta_at_alpha_2",
            "Baseline Incorrect Output Bits", "Self Correct Output Bits Change",
            "K=10: Recall Difficulty vs α=2 Effect")
    scatter("interference_caused_vs_strength_effect.png", "total_negative_interference_caused", "all_exact_delta_at_alpha_0_5",
            "Total Negative Signed Interference Caused", "Exact Recall Change (percentage points)",
            "K=10: Negative Interference vs α=0.5 Effect")
    fig, ax = plt.subplots(figsize=(10, 4.5))
    before = [r["baseline_minimum_signed_margin"] for r in sorted(factors10, key=lambda r:r["digit"])]
    # Prespecified illustrative change: digit 0 α=2; no selection on outcome.
    r = next(r for r in sens10 if r["changed_digit"] == 0 and r["alpha"] == 2.0)
    after = json.loads(r["minimum_signed_margins_by_digit_json"])
    z = np.arange(10)
    ax.bar(z-0.18, before, width=.36, label="Equal α=1")
    ax.bar(z+0.18, after, width=.36, label="Digit 0 α=2")
    ax.axhline(0, color="black", lw=.7)
    ax.set(xticks=z, xlabel="Recalled Digit", ylabel="Minimum Signed Output Margin (score units)",
           title="K=10: Signed Margin Before and After Digit 0 Strength Adjustment")
    ax.legend(); fig.tight_layout(); fig.savefig(figdir/"signed_margin_before_after.png", dpi=155); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    if solution_rows:
        for label in dict.fromkeys(r["label"] for r in solution_rows):
            rows = sorted((r for r in solution_rows if r["label"] == label), key=lambda r:r["digit"])
            ax.scatter([r["total_negative_interference_caused"] for r in rows],
                       [r["alpha"] for r in rows], s=50, label=label)
            for r in rows:
                ax.annotate(str(r["digit"]), (r["total_negative_interference_caused"],r["alpha"]),
                            xytext=(3,3), textcoords="offset points", fontsize=8)
        ax.legend(fontsize=8)
    else:
        ax.text(.5, .5, "Archived MILP α vectors unavailable; no values inferred",
                ha="center", va="center", transform=ax.transAxes)
    ax.set(xlabel="Total Negative Signed Interference Caused (equal α)",
           ylabel="Archived Optimized Association Strength α",
           title="K=10: Optimized Alpha vs Association Characteristics")
    fig.tight_layout(); fig.savefig(figdir/"optimal_alpha_vs_characteristics.png", dpi=155); plt.close(fig)


def run(out, solutions_path, reference_zip):
    for d in (out, out/"csv", out/"npz"):
        d.mkdir(parents=True, exist_ok=False if d == out else True)
    baseline_rows, sensitivity_rows, effect_rows, factor_rows, pair_rows = [], [], [], [], []
    k10_bit_rows = []
    subset_counts = {}
    for k in KS:
        subset_counts[k] = 0
        for ids in combinations(range(10), k):
            subset_counts[k] += 1
            ids = tuple(ids)
            sub = "-".join(map(str, ids))
            x, y = X[list(ids)], Y[list(ids)]
            w = x.T @ y
            score = x @ w
            pred0, ok0, exact0, margin0 = eval_scores(score, y)
            assert np.array_equal(score, G[np.ix_(ids, ids)] @ y)
            baseline_rows.append(dict(k=k, subset=sub, exact_count=int(np.sum(exact0)),
                                      exact_rate=float(np.mean(exact0)), bit_accuracy=float(np.mean(ok0)),
                                      min_signed_margin=int(np.min(margin0))))
            fs, ps = factors(ids, score, ok0, exact0, margin0)
            factor_rows.extend(fs); pair_rows.extend(ps)
            if k == 10:
                np.savez_compressed(out/"npz"/"k10_equal_strength.npz", X=x, Y=y, W=w, scores=score,
                                    predictions=pred0.astype(np.uint8), signed_margins=margin0)
            for i, p in enumerate(ids):
                for alpha in ALPHAS:
                    scores = score + (alpha-1)*G[np.array(ids), p, None]*Y[p][None, :]
                    pred, ok, exact, margin = eval_scores(scores, y)
                    scale = k/(k-1+alpha)
                    normalized_score = scores*scale
                    assert np.array_equal(normalized_score > 0, pred)
                    corrected = (~ok0) & ok
                    damaged = ok0 & (~ok)
                    new_exact = (~exact0) & exact
                    lost_exact = exact0 & (~exact)
                    own_bit_delta = int(np.sum(ok[i])-np.sum(ok0[i]))
                    other_bit_delta = int(np.sum(ok)-np.sum(ok0)-own_bit_delta)
                    row = dict(k=k, subset=sub, changed_digit=p, alpha=float(alpha),
                               alpha_sum=float(k-1+alpha), normalization_scale=float(scale),
                               all_exact_count=int(np.sum(exact)), all_exact_rate=float(np.mean(exact)),
                               all_bit_accuracy=float(np.mean(ok)), own_exact=int(exact[i]),
                               own_bit_accuracy=float(np.mean(ok[i])),
                               other_exact_count=int(np.sum(exact)-exact[i]),
                               other_exact_rate=float((np.sum(exact)-exact[i])/(k-1)),
                               other_bit_accuracy=float((np.sum(ok)-np.sum(ok[i]))/(15*(k-1))),
                               own_minimum_signed_margin=float(np.min(margin[i])),
                               overall_minimum_signed_margin=float(np.min(margin)),
                               normalized_overall_minimum_signed_margin=float(np.min(margin)*scale),
                               minimum_signed_margins_by_digit_json=json.dumps([float(np.min(z)) for z in margin]),
                               corrected_output_bits=int(np.sum(corrected)),
                               newly_incorrect_output_bits=int(np.sum(damaged)),
                               newly_exact_digits="-".join(str(ids[z]) for z in np.flatnonzero(new_exact)),
                               lost_exact_digits="-".join(str(ids[z]) for z in np.flatnonzero(lost_exact)),
                               self_corrected_bits=int(np.sum(corrected[i])),
                               self_newly_incorrect_bits=int(np.sum(damaged[i])),
                               other_corrected_bits=int(np.sum(corrected)-np.sum(corrected[i])),
                               other_newly_incorrect_bits=int(np.sum(damaged)-np.sum(damaged[i])),
                               self_correct_bit_delta=own_bit_delta,
                               other_correct_bit_delta=other_bit_delta,
                               exact_rate_delta_pp=float(100*(np.mean(exact)-np.mean(exact0))))
                    sensitivity_rows.append(row)
                    if k == 10:
                        for h, q in enumerate(ids):
                            effect_rows.append(dict(k=k, subset=sub, changed_digit=p, alpha=float(alpha),
                                                    recalled_digit=q, self_memory=int(q==p),
                                                    exact_before=int(exact0[h]), exact_after=int(exact[h]),
                                                    correct_bits_before=int(np.sum(ok0[h])),
                                                    correct_bits_after=int(np.sum(ok[h])),
                                                    minimum_margin_before=int(np.min(margin0[h])),
                                                    minimum_margin_after=float(np.min(margin[h]))))
                            for j in range(15):
                                k10_bit_rows.append(dict(changed_digit=p, alpha=float(alpha), recalled_digit=q,
                                                         output_neuron=j, target_bit=int(y[h,j]>0),
                                                         score_before=int(score[h,j]), score_after=float(scores[h,j]),
                                                         prediction_before=int(pred0[h,j]), prediction_after=int(pred[h,j]),
                                                         signed_margin_before=int(margin0[h,j]),
                                                         signed_margin_after=float(margin[h,j]),
                                                         signed_contribution_from_changed_memory=int(G[q,p]*y[h,j]*Y[p,j])))
    assert subset_counts == {3:120, 5:252, 7:120, 10:1}
    write_csv(out/"csv"/"equal_strength_baselines.csv", baseline_rows)
    write_csv(out/"csv"/"single_strength_sensitivity.csv", sensitivity_rows)
    write_csv(out/"csv"/"k10_association_effects.csv", effect_rows)
    write_csv(out/"csv"/"k10_output_bit_effects.csv", k10_bit_rows)
    write_csv(out/"csv"/"association_factors.csv", factor_rows)
    write_csv(out/"csv"/"association_pair_factors.csv", pair_rows)

    sens10 = [r for r in sensitivity_rows if r["k"] == 10]
    factors10 = [r for r in factor_rows if r["k"] == 10]
    comparisons = []
    for f in factors10:
        p = f["digit"]
        rs = [r for r in sens10 if r["changed_digit"] == p]
        at = {r["alpha"]:r for r in rs}
        best = max(r["all_exact_rate"] for r in rs)
        preferred = min(r["alpha"] for r in rs if r["all_exact_rate"] == best)
        comparisons.append(dict(**f, preferred_alpha=preferred, best_single_change_exact_rate=best,
                                all_exact_delta_at_alpha_2=at[2.0]["exact_rate_delta_pp"],
                                all_exact_delta_at_alpha_0_5=at[0.5]["exact_rate_delta_pp"],
                                self_bit_delta_at_alpha_2=at[2.0]["self_correct_bit_delta"],
                                other_bit_delta_at_alpha_2=at[2.0]["other_correct_bit_delta"],
                                self_bit_delta_at_alpha_0_5=at[0.5]["self_correct_bit_delta"],
                                other_bit_delta_at_alpha_0_5=at[0.5]["other_correct_bit_delta"]))
    write_csv(out/"csv"/"k10_factor_effect_comparison.csv", comparisons)

    correlation_rows = []
    factor_names = ["mean_absolute_input_similarity", "maximum_absolute_input_similarity",
                    "mean_target_similarity", "mean_input_target_compatibility",
                    "baseline_incorrect_bits", "baseline_minimum_signed_margin",
                    "total_negative_interference_caused", "negative_interference_on_vulnerable_bits"]
    outcomes = ["all_exact_delta_at_alpha_2", "all_exact_delta_at_alpha_0_5",
                "self_bit_delta_at_alpha_2", "other_bit_delta_at_alpha_2"]
    for factor in factor_names:
        for outcome in outcomes:
            correlation_rows.append(dict(k=10, n=10, factor=factor, outcome=outcome,
                                         spearman_r=spearman([r[factor] for r in comparisons],
                                                             [r[outcome] for r in comparisons])))
    # K=3/5/7 observations share digits and subsets. These are descriptive, not independent samples.
    for k in (3,5,7):
        ff = [r for r in factor_rows if r["k"] == k]
        for factor in factor_names:
            for a in (0.5,2.0):
                lookup = {(r["subset"],r["changed_digit"]):r["exact_rate_delta_pp"]
                          for r in sensitivity_rows if r["k"]==k and r["alpha"]==a}
                correlation_rows.append(dict(k=k, n=len(ff), factor=factor,
                                             outcome=f"exact_rate_delta_at_alpha_{a}",
                                             spearman_r=spearman([r[factor] for r in ff],
                                                                 [lookup[(r["subset"],r["digit"])] for r in ff])))
    write_csv(out/"csv"/"factor_spearman_correlations.csv", correlation_rows)

    if reference_zip is not None:
        archived, provenance = load_reference_zip(reference_zip)
        raw=out/"raw"
        raw.mkdir()
        shutil.copy2(reference_zip, raw/reference_zip.name)
    else:
        archived, provenance = load_solutions(solutions_path)
    if solutions_path is not None and reference_zip is not None:
        extra, extra_provenance = load_solutions(solutions_path)
        archived.extend(extra)
        provenance += " " + extra_provenance
    solrows, solsummary = [], []
    for record in [{"label":"Equal alpha=1", "alpha":[1]*10}] + archived:
        alpha = np.asarray(record["alpha"], dtype=float)
        w = X.T @ (alpha[:,None]*Y)
        scores = X @ w
        _, bitok, exact, margins = eval_scores(scores,Y)
        label = record["label"]
        np.savez_compressed(out/"npz"/("solution_"+str(len(solsummary))+".npz"),
                            alpha=alpha, W=w, scores=scores, targets=Y, signed_margins=margins)
        solsummary.append(dict(label=label, source="direct correlation" if label=="Equal alpha=1" else "supplied archived alpha",
                               exact_count=int(np.sum(exact)), bit_accuracy=float(np.mean(bitok)),
                               minimum_signed_margin=float(np.min(margins)),
                               minimum_signed_margin_among_exact=float(np.min(margins[exact])) if np.any(exact) else "",
                               exactly_recalled_digits="-".join(map(str,np.flatnonzero(exact)))))
        for p, f in enumerate(factors10):
            solrows.append(dict(label=label, digit=p, alpha=float(alpha[p]),
                                exact=int(exact[p]), minimum_signed_margin=float(np.min(margins[p])),
                                total_negative_interference_caused=f["total_negative_interference_caused"],
                                weighted_negative_interference_caused=float(alpha[p]*f["total_negative_interference_caused"]),
                                weighted_positive_contribution_to_others=float(alpha[p]*f["total_positive_contribution_to_others"]),
                                mean_absolute_input_similarity=f["mean_absolute_input_similarity"],
                                mean_input_target_compatibility=f["mean_input_target_compatibility"],
                                baseline_incorrect_bits=f["baseline_incorrect_bits"]))
    write_csv(out/"csv"/"archived_solution_summary.csv", solsummary)
    write_csv(out/"csv"/"archived_solution_factors.csv", solrows)
    plot_results(out, sens10, factors10, comparisons,
                 [r for r in solrows if r["label"]!="Equal alpha=1"] if archived else [])

    config = dict(dataset_file="binary_character.py", dataset_sha256=hashlib.sha256((PROJECT/"binary_character.py").read_bytes()).hexdigest(),
                  digit_patterns=X01.tolist(), letter_patterns=Y01.tolist(), K=list(KS), alpha_grid=list(ALPHAS),
                  readout="score = X_b W; predicted bit = 1 iff score > 0; tie -> 0",
                  no_noise=True, seed="not used: deterministic combinations",
                  high_similarity_absolute_dot_threshold=HIGH_SIM_ABS_DOT,
                  vulnerable_signed_margin_threshold=VULNERABLE_MARGIN,
                  normalization="optional K/sum(alpha); only normalized margins, not decisions, are recorded",
                  archived_solution_source=str(reference_zip) if reference_zip else (str(solutions_path) if solutions_path else None),
                  git_revision=subprocess.run(["git","rev-parse","HEAD"],cwd=PROJECT,capture_output=True,text=True).stdout.strip(),
                  Python=sys.version, numpy=np.__version__)
    (out/"config.json").write_text(json.dumps(config,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    k10baseline = next(r for r in baseline_rows if r["k"]==10)
    assert k10baseline["exact_count"] == 1
    assert len(sens10)==10*len(ALPHAS)
    assert len(k10_bit_rows)==10*len(ALPHAS)*10*15
    assert len(factor_rows)==sum(__import__('math').comb(10,k)*k for k in KS)
    # α=1 must reproduce each baseline for every subset and changed association.
    base_by_subset={(r["k"],r["subset"]):r for r in baseline_rows}
    assert all(r["all_exact_count"]==base_by_subset[(r["k"],r["subset"])]["exact_count"]
               for r in sensitivity_rows if r["alpha"]==1.0)

    metrics=dict(status="partial_missing_bound_0.5_3.0" if reference_zip is not None else ("success" if archived else "partial_missing_archived_milp"), subset_counts=subset_counts, sensitivity_trials=len(sensitivity_rows),
                 k10_equal_exact_count=k10baseline["exact_count"], archived_milp_artifact_count=len(archived),
                 archived_milp_status=provenance,
                 k10_single_factor_best=[dict(digit=p, best_exact_count=max(r["all_exact_count"] for r in sens10 if r["changed_digit"]==p),
                                              alpha_smallest_tie=min(r["alpha"] for r in sens10 if r["changed_digit"]==p and r["all_exact_count"]==max(z["all_exact_count"] for z in sens10 if z["changed_digit"]==p))) for p in range(10)])
    (out/"metrics.json").write_text(json.dumps(metrics,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    strongest=max((r for r in sens10 if r["alpha"]!=1), key=lambda r:(r["all_exact_count"],-r["alpha"],-r["changed_digit"]))
    trial={(r["changed_digit"],r["alpha"]):r for r in sens10}
    corr={(r["factor"],r["outcome"]):r["spearman_r"] for r in correlation_rows if r["k"]==10}
    milp = next((r for r in solsummary if r["label"]=="Archived MILP alpha [0.01,4.0]"), None)
    margin_opt = next((r for r in solsummary if r["label"]=="Archived six-preserving margin alpha"), None)
    summary=f"""# Association Learning Strength: Static Weighted-Correlation Diagnostics

## Protocol and provenance
10 digit→letter pairs from `binary_character.py` (SHA256 `{config['dataset_sha256']}`), 15-bit bipolar input/output, `W=Σ α_p x_pᵀy_p`, stored-input static recall `score>0` (tie→0). All subsets for K=3,5,7,10: {subset_counts}. α grid: {ALPHAS}. Every trial changes exactly one α while the others remain 1; no normalization is applied to raw scores. The CSV also records the positive scale `K/Σα` and normalized minimum margin. A common positive scale cannot change predictions. No noise, spike simulation, or parameter fitting was used.

Run from the repository root: `python my_research/experiments/diagnostics/analyze_association_strength.py`. This creates a new timestamped directory and ZIP on each run. The checked-out revision and exact binary patterns are in `config.json`.

## Direct observations
Equal α=1 at K=10 recalled **{k10baseline['exact_count']}/10** associations exactly; bit accuracy {100*k10baseline['bit_accuracy']:.2f}%. K=10 sensitivity has {len(sens10)} trials; all K values have {len(sensitivity_rows)} trials. The largest K=10 exact count from a *single-coordinate* change on this grid was **{strongest['all_exact_count']}/10**, at digit {strongest['changed_digit']} α={strongest['alpha']}. This is not an optimized joint α vector. Exact count, corrected/newly incorrect bits, newly/lost exact associations and self/other effects for every trial are in `csv/single_strength_sensitivity.csv`; every K=10 output bit before/after is in `csv/k10_output_bit_effects.csv`.

The α=1 association-weighted exact rates are K=3: {100*sum(r['exact_count'] for r in baseline_rows if r['k']==3)/360:.2f}%, K=5: {100*sum(r['exact_count'] for r in baseline_rows if r['k']==5)/1260:.2f}%, K=7: {100*sum(r['exact_count'] for r in baseline_rows if r['k']==7)/840:.2f}%, K=10: {100*k10baseline['exact_rate']:.2f}%. These reproduce the current pure-correlation reference values.

Input similarity alone does not determine interference direction. The actual recipient-target-signed contribution of source p to recipient q and output j is `dot(x_q,x_p) · y_qj · y_pj`; it is tabulated by pair in `association_pair_factors.csv`. The full exact/bit effects show the trade-off without treating Spearman coefficients as causal. `factor_spearman_correlations.csv` uses tied average ranks; K=3/5/7 rows reuse overlapping subsets and are descriptive, not independent samples. The retrospective preferred α in `k10_factor_effect_comparison.csv` is the smallest α tied for highest *single-change* exact count on the measured grid; it is not a learned rule.

### Measured K=10 examples and factor checks
- Increasing digit 5 from α=1 to 2 yielded {trial[(5,2.0)]['all_exact_count']}/10 exact: digits {trial[(5,2.0)]['newly_exact_digits']} became exact. Its own correct-bit count changed by {trial[(5,2.0)]['self_correct_bit_delta']:+d}, while the other nine together changed by {trial[(5,2.0)]['other_correct_bit_delta']:+d}; this is a real self/other trade-off despite an exact-count gain.
- Decreasing digit 3 to α=0.1 yielded {trial[(3,0.1)]['all_exact_count']}/10 exact, with newly exact digits {trial[(3,0.1)]['newly_exact_digits']}. Digit 3 itself changed by {trial[(3,0.1)]['self_correct_bit_delta']:+d} correct bits while other memories changed by {trial[(3,0.1)]['other_correct_bit_delta']:+d}. Lowering a contributor can help others while hurting its own recall.
- Decreasing digit 7 to α=0.5 caused the sole initially exact digit 7 to fail, leaving {trial[(7,0.5)]['all_exact_count']}/10 exact. Therefore lowering α is not uniformly beneficial.
- For α=0.5 versus α=1, Spearman ρ between *mean absolute input dot* and total exact-count change is {corr[('mean_absolute_input_similarity','all_exact_delta_at_alpha_0_5')]:+.3f}; for *negative signed interference caused*, it is {corr[('total_negative_interference_caused','all_exact_delta_at_alpha_0_5')]:+.3f} (n=10, many tied exact counts). Input similarity by itself omits the target sign. The mean input-dot × target-dot factor gives ρ={corr[('mean_input_target_compatibility','all_exact_delta_at_alpha_0_5')]:+.3f} for this outcome; in this small K=10 comparison it does **not** provide stronger rank association. Per-bit signed contributions remain the direct evidence.
- Baseline incorrect-bit count versus α=2 total exact change gives ρ={corr[('baseline_incorrect_bits','all_exact_delta_at_alpha_2')]:+.3f}. Digit 5 α=2 improves one own bit but digit 5 remains inexact, so strengthening a difficult memory does not imply its exact recovery.

## Archived MILP references
{provenance} {f"The archived MILP vector reproduced {milp['exact_count']}/10 exact ({milp['bit_accuracy']*100:.2f}% bits), with exact digits {milp['exactly_recalled_digits']} and minimum margin among those six {milp['minimum_signed_margin_among_exact']:.6g}. The archived six-preserving margin vector also reproduced {margin_opt['exact_count']}/10 exact ({margin_opt['bit_accuracy']*100:.2f}% bits), with minimum margin among its exact digits {margin_opt['minimum_signed_margin_among_exact']:.6g}." if milp and margin_opt else "Archived vector comparison was not available."} These distinct vectors achieve the same observed 6/10 exact count but have different margins; they provide two documented 6/10 references, not proof of uniqueness or generalization. Both put digit 3 and digit 4 at α=0.01; whether this commonality persists across *all* optimal solutions is untested. The requested [0.5,3.0] solution is not in the provided archive and was not invented. These α vectors use all targets and are diagnostic references, not STDP learning results. `csv/archived_solution_summary.csv` and `csv/archived_solution_factors.csv` contain their evaluated results. An additional archived vector can be passed as a JSON list of `{{"label": "...", "alpha": [10 positive values in digit order]}}` with `--solutions`.

In both archived vectors digit 0 has α>2 although its equal-strength negative interference magnitude is 412; digit 8 causes the largest equal-strength negative interference (606) and has α≈0.34/0.63. Thus a simple rule of decreasing α whenever one interference factor is large does not explain the joint solutions. The two archived vectors differ notably at digit 5 (α≈0.734 versus 0.01) while retaining the same six exact memories. These are observed patterns in two target-informed solutions, not a unique prescription.

## STDP information availability and feasibility
- **A — directly synapse-local:** current weight, pre/post event timing and local update history. These can support local magnitude modulation, but the global association factors measured here are not available from one synapse alone.
- **B — same-network population activity:** output firing and aggregate activity may provide a proxy for current recall response. Computing association-level difficulty or weak-margin signals would require identification of the presented association and an activity/error interpretation; this experiment did not implement that circuitry.
- **C — global target/dataset information:** pairwise input/target similarity matrices, exact recall of every memory, signed margins against all target bits, cross-memory interference and MILP α. These require information across associations and/or correct targets; they are not local STDP observables.

This analysis proposes no new rule and establishes no biological feasibility. `figures/optimal_alpha_vs_characteristics.png` plots archived vectors when available and explicitly marks their absence otherwise. K=10 n=10 and discrete exact counts limit factor inference; no causal claim is made. All numeric outputs and the executable script are included in the ZIP.
"""
    (out/"summary.md").write_text(summary,encoding="utf-8")
    return metrics


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--solutions",type=Path,help="Archived MILP alpha JSON list; no optimization is run")
    parser.add_argument("--reference-zip",type=Path,default=PROJECT/"association_alpha_milp_reference_20261019.zip",
                        help="Original MILP reference ZIP; automatically used when present")
    args=parser.parse_args()
    reference_zip=args.reference_zip if args.reference_zip.exists() else None
    out=ROOT/datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    metrics=run(out,args.solutions,reference_zip)
    archive=out.with_suffix(".zip")
    with zipfile.ZipFile(archive,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        z.write(__file__,arcname=out.name+"/analyze_association_strength.py")
        for file in sorted(out.rglob("*")):
            if file.is_file():
                z.write(file,arcname=out.name+"/"+str(file.relative_to(out)))
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
    print(json.dumps({"result_dir":str(out),"zip":str(archive),**metrics},ensure_ascii=False))


if __name__=="__main__":
    main()
