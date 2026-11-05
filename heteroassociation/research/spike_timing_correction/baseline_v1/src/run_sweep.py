"""CS-duration × learning-rate sweep of the validated continuous-state LIF/STDP model.

The model is an isolated copy of the validated lambda=1 source. Only CS_END_TICK,
LAMBDA, and removal of the current cap affect dynamics; added counters are read-only.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
import time
import types
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "scripts/sweep_model.py"
ENDS = (4.0, 4.2, 4.4, 4.5)
RATES = (1.0, 0.5, 0.25, 0.1, 0.05)
DATASETS = ("Original", "Modified")


def model_module():
    if "numba" not in sys.modules and importlib.util.find_spec("numba") is None:
        shim = types.ModuleType("numba")
        shim.njit = lambda function: function
        sys.modules["numba"] = shim
    spec = importlib.util.spec_from_file_location("sweep_model", MODEL_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def same_condition_id(dataset, end, rate):
    return f"{dataset.lower()}_end{end:.1f}_lambda{rate:g}"


def recall(model, x, y, w, voltage, refractory, next_tick):
    """Teacher-free sequential copy-state readout from the validated model."""
    v, r = voltage.copy(), refractory.copy()
    prediction = np.zeros_like(y)
    zero_target = np.zeros(15, dtype=np.int64)
    for p in range(10):
        tick = next_tick + p * (model.TRIAL_TICKS + model.GAP_TICKS)
        prediction[p], counts, post_ticks = model.lif_trial(
            x[p], w, zero_target, v, r, tick, False
        )
        assert not np.any(post_ticks[:, 0] == 0)
        if model.GAP_TICKS:
            model.zero_current_gap(v, r, tick + model.TRIAL_TICKS, model.GAP_TICKS)
    return (int(np.all(prediction == y, axis=1).sum()),
            float(np.mean(prediction == y)),
            int(np.sum((prediction == 1) & (y == 0))),
            int(np.sum((prediction == 0) & (y == 1))))


def simulate(dataset, end, rate, epochs=1000, current_cap=None, save_events=True):
    model = model_module()
    model.CS_END_TICK = int(round(end / model.DT))
    model.LAMBDA = rate
    model.CAP = current_cap
    assert end in ENDS and rate in RATES
    assert model.DT == 0.1 and model.GAP_TICKS == 50
    assert model.TRIAL_TICKS == 101 and model.TEACHER_I == 30.0
    assert abs(model.window(-1) - .05) < 1e-7
    assert abs(model.window(3) + .05) < 1e-7
    x, y, orders = model.load_data(dataset)
    assert x.shape == y.shape == (10, 15) and orders.shape == (1000, 10)
    w = np.zeros((15, 15))
    voltage = np.zeros(15)
    refractory = np.full(15, -1000000000, dtype=np.int64)
    stdp = model.ContinuousAllPairs()
    tick = trial_number = 0
    if save_events:
        events = np.full((epochs, 10, 15, 12), -1, dtype=np.int16)
        counts_log = np.zeros((epochs, 10, 15), dtype=np.uint8)
        ltp_log = np.zeros((epochs, 10), dtype=np.float64)
        ltd_log = np.zeros_like(ltp_log)
        residual_11_log = np.zeros_like(ltp_log)
    history = []
    teacher_expected = teacher_observed = 0
    natural_total = extra_natural = target0_false_spikes = 0
    before_input_spikes = positive_window_natural = 0
    natural_times = []
    fp_times = []
    target1_times = []
    start = time.time()
    for ep in range(epochs):
        for position, association in enumerate(orders[ep]):
            trial_number += 1
            prediction, counts, post_ticks = model.lif_trial(
                x[association], w, y[association], voltage, refractory, tick, True
            )
            if save_events:
                events[ep, position] = post_ticks
                counts_log[ep, position] = counts
            for j in range(15):
                post = post_ticks[j, :counts[j]]
                teacher_expected += int(y[association, j])
                teacher_observed += int(0 in post)
                if y[association, j] == 0 and 0 in post:
                    raise AssertionError("Target=0 generated teacher spike")
                natural = post[post > 0]
                assert int(bool(natural.size)) == prediction[j]
                natural_total += natural.size
                extra_natural += max(0, natural.size - 1)
                before_input_spikes += int(np.sum(natural < 10))
                for local_tick in natural:
                    ms = float(local_tick * model.DT)
                    natural_times.append(ms)
                    if y[association, j]:
                        target1_times.append(ms)
                    else:
                        target0_false_spikes += 1
                        fp_times.append(ms)
                    if model.window((local_tick - 10) * model.DT) > 0:
                        positive_window_natural += 1
            delta = stdp.process(x[association], counts, post_ticks, tick, trial_number)
            if save_events:
                ltp_log[ep, position] = stdp.last_positive_dw
                ltd_log[ep, position] = stdp.last_negative_dw
                active = int(np.sum(x[association]))
                # Within-trial 11 cancellation; cross-trial pairs remain in delta.
                residual = 0.0
                for j in range(15):
                    if y[association, j] and prediction[j]:
                        output_ticks = post_ticks[j, :counts[j]]
                        residual += active * rate * sum(
                            model.window((int(k) - 10) * model.DT) for k in output_ticks
                        )
                residual_11_log[ep, position] = residual
            w += delta
            if not np.isfinite(w).all():
                raise FloatingPointError(f"Nonfinite weights: {dataset}, {end}, {rate}, {ep+1}")
            if model.GAP_TICKS:
                model.zero_current_gap(voltage, refractory, tick + model.TRIAL_TICKS,
                                       model.GAP_TICKS)
            tick += model.TRIAL_TICKS + model.GAP_TICKS
        exact, bit, fp, fn = recall(model, x, y, w, voltage, refractory, tick)
        if current_cap is not None and epochs == 1000:
            # Regression path: independent and sequential probes must agree.
            independent = model.recall_independent(x, y, w)
            assert exact == independent
        history.append({"dataset": dataset, "cs_end_ms": end, "lambda": rate,
                        "epoch": ep + 1, "exact_count": exact,
                        "exact_accuracy_percent": exact * 10,
                        "bit_accuracy_percent": bit * 100,
                        "false_positive_bits": fp, "false_negative_bits": fn,
                        "weight_norm": float(np.linalg.norm(w))})
    full = [int(r["epoch"]) for r in history if r["exact_count"] == 10]
    first = min(full) if full else None
    sustained = (next((int(r["epoch"]) for r in reversed(history)
                       if r["exact_count"] != 10), 0) + 1
                 if history[-1]["exact_count"] == 10 else None)
    summary = {
        "dataset": dataset, "cs_end_ms": end, "lambda": rate,
        "epochs": epochs, "first_10_epoch": first,
        "sustained_10_start_epoch": sustained,
        "final_exact_count": history[-1]["exact_count"],
        "final_exact_accuracy_percent": history[-1]["exact_accuracy_percent"],
        "final_bit_accuracy_percent": history[-1]["bit_accuracy_percent"],
        "last_100_full_epochs": sum(r["exact_count"] == 10 for r in history[-100:]),
        "teacher_spikes_expected": teacher_expected,
        "teacher_spikes_observed": teacher_observed,
        "natural_spikes": natural_total, "extra_natural_spikes": extra_natural,
        "natural_spikes_before_input": before_input_spikes,
        "natural_spikes_in_positive_window": positive_window_natural,
        "target0_false_positive_spikes": target0_false_spikes,
        "cross_association_pairs": stdp.cross_count,
        "cross_association_abs_update": stdp.cross_abs,
        "final_weight_min": float(np.min(w)),
        "final_weight_max": float(np.max(w)),
        "final_weight_norm": float(np.linalg.norm(w)),
        "runtime_seconds": time.time() - start,
    }
    timing = {key: summary[key] for key in (
        "dataset", "cs_end_ms", "lambda", "teacher_spikes_expected",
        "teacher_spikes_observed", "natural_spikes", "extra_natural_spikes",
        "natural_spikes_before_input", "natural_spikes_in_positive_window",
        "target0_false_positive_spikes", "cross_association_pairs",
        "cross_association_abs_update")}
    timing.update({
        "mean_natural_spike_time_ms": float(np.mean(natural_times)) if natural_times else None,
        "min_natural_spike_time_ms": float(min(natural_times)) if natural_times else None,
        "max_natural_spike_time_ms": float(max(natural_times)) if natural_times else None,
        "mean_target1_spike_time_ms": float(np.mean(target1_times)) if target1_times else None,
        "mean_false_positive_spike_time_ms": float(np.mean(fp_times)) if fp_times else None,
        "gross_LTP": float(ltp_log.sum()) if save_events else None,
        "gross_LTD": float(ltd_log.sum()) if save_events else None,
        "net_update": float(ltp_log.sum() + ltd_log.sum()) if save_events else None,
        "absolute_11_residual_update": float(np.abs(residual_11_log).sum()) if save_events else None,
    })
    return model, x, y, w, history, summary, timing, (
        (events, counts_log, ltp_log, ltd_log, residual_11_log) if save_events else None
    )


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def run_condition(args):
    dataset, end, rate = args
    model, x, y, w, history, summary, timing, logs = simulate(dataset, end, rate)
    key = same_condition_id(dataset, end, rate)
    events, counts_log, ltp, ltd, residual = logs
    np.savez_compressed(ROOT / "npz" / f"{key}.npz",
                        input_binary=x, target_binary=y,
                        presentation_orders=model.load_data(dataset)[2],
                        output_spike_ticks=events, output_spike_count=counts_log,
                        gross_LTP_by_presentation=ltp,
                        gross_LTD_by_presentation=ltd,
                        residual_11_by_presentation=residual,
                        final_weight=w, dt_ms=model.DT,
                        cs_end_ms=end, learning_rate=rate)
    write_csv(ROOT / "csv" / f"epoch_{key}.csv", history)
    (ROOT / "csv" / f"condition_{key}.json").write_text(
        json.dumps({"summary": summary, "timing": timing}, indent=2) + "\n")
    print(f"FINISHED {key}: final={summary['final_exact_count']}/10, "
          f"first={summary['first_10_epoch']}, sustained={summary['sustained_10_start_epoch']}, "
          f"seconds={summary['runtime_seconds']:.1f}", flush=True)
    return summary, timing


def baseline_validation():
    reference = Path(__file__).resolve().parents[3] / "research/continuous_stdp_heteroassociation/results/csv/epoch_history.csv"
    assert reference.is_file(), reference
    with reference.open() as file:
        prior = [r for r in csv.DictReader(file) if r["model"] == "Baseline"]
    checks = []
    for dataset in DATASETS:
        model, x, y, w, history, summary, timing, _ = simulate(
            dataset, 4.0, 1.0, epochs=1000,
            current_cap=1.282590743781507, save_events=False
        )
        previous = [r for r in prior if r["dataset"] == dataset]
        assert len(history) == len(previous) == 1000
        assert all(int(a["exact_count"]) == int(b["exact_count_continuous_recall"])
                   and abs(a["bit_accuracy_percent"] / 100 -
                           float(b["bit_accuracy_continuous_recall"])) < 1e-12
                   for a, b in zip(history, previous))
        assert summary["final_exact_count"] == 10
        checks.append({"dataset": dataset, "condition": "archived capped lambda=1",
                       "epoch_accuracy_matches": 1000, "teacher_spike_reliability":
                       f"{summary['teacher_spikes_observed']}/{summary['teacher_spikes_expected']}",
                       "final_exact_count": summary["final_exact_count"]})
    # A capped near-threshold current proves that k=40 is a true threshold
    # crossing at the END of the [3.9,4.0) Euler interval, not at its start.
    model = model_module()
    model.CAP = 1.282590743781507
    model.CS_END_TICK = 40
    x, y, _ = model.load_data("Original")
    test_w = np.full((15, 15), model.CAP / int(x[0].sum()))
    v = np.zeros(15)
    r = np.full(15, -1000000000, dtype=np.int64)
    _, counts, ticks = model.lif_trial(x[0], test_w, y[0], v, r, 0, True)
    assert all(40 in ticks[j, :counts[j]] for j in range(15))
    checks.append({"dataset": "Original", "condition": "Euler boundary",
                   "epoch_accuracy_matches": "", "teacher_spike_reliability": "",
                   "final_exact_count": "spike at 4.0 ms from [3.9,4.0)"})
    write_csv(ROOT / "csv/baseline_validation.csv", checks)
    print("PASS baseline: both datasets, 1000/1000 epoch parity; 4.0-ms end-of-step spike", flush=True)


def aggregate_existing():
    summaries, timings, epochs, timing_histogram = [], [], [], []
    for dataset in DATASETS:
        for end in ENDS:
            for rate in RATES:
                key = same_condition_id(dataset, end, rate)
                condition_csv = ROOT / "csv" / f"epoch_{key}.csv"
                condition_npz = ROOT / "npz" / f"{key}.npz"
                condition_json = ROOT / "csv" / f"condition_{key}.json"
                if not (condition_csv.is_file() and condition_npz.is_file() and condition_json.is_file()):
                    continue
                with condition_csv.open() as file:
                    rows = list(csv.DictReader(file))
                assert len(rows) == 1000
                epochs.extend(rows)
                record = json.loads(condition_json.read_text())
                with np.load(condition_npz) as z:
                    exact = np.array([int(row["exact_count"]) for row in rows])
                    first = int(np.flatnonzero(exact == 10)[0] + 1) if np.any(exact == 10) else None
                    stable = int(np.flatnonzero(exact != 10)[-1] + 2) if exact[-1] == 10 and np.any(exact != 10) else (1 if np.all(exact == 10) else None)
                    spike = z["output_spike_ticks"]
                    y = z["target_binary"]
                    order = z["presentation_orders"]
                    teacher = spike[:, :, :, 0] == 0
                    target_at_position = y[order]
                    assert np.array_equal(teacher, target_at_position == 1)
                    natural = spike[spike > 0]
                    natural_count = (spike > 0).sum(axis=-1)
                    first_ticks = np.where(spike > 0, spike, 1000).min(axis=-1)
                    first_mask = first_ticks < 1000
                    first_ms = first_ticks[first_mask] * .1
                    target1_first = first_ticks[first_mask & (target_at_position == 1)] * .1
                    false_positive_first = first_ticks[first_mask & (target_at_position == 0)] * .1
                    summary = record["summary"]
                    assert summary["first_10_epoch"] == first
                    assert summary["sustained_10_start_epoch"] == stable
                    assert summary["final_exact_count"] == int(exact[-1])
                    assert summary["teacher_spikes_expected"] == int(target_at_position.sum())
                    assert summary["teacher_spikes_observed"] == int(teacher.sum())
                    assert summary["natural_spikes"] == int(natural.size)
                    summaries.append(summary)
                    timing = record["timing"]
                    timing.update({
                        "first_natural_count": int(first_mask.sum()),
                        "mean_first_natural_ms": float(np.mean(first_ms)) if first_ms.size else None,
                        "p05_first_natural_ms": float(np.quantile(first_ms, .05)) if first_ms.size else None,
                        "p95_first_natural_ms": float(np.quantile(first_ms, .95)) if first_ms.size else None,
                        "mean_target1_first_ms": float(np.mean(target1_first)) if target1_first.size else None,
                        "mean_false_positive_first_ms": float(np.mean(false_positive_first)) if false_positive_first.size else None,
                        "multiple_output_trials": int(np.sum(natural_count > 1)),
                        "natural_spikes_before_cs_start": int(np.sum((spike > 0) & (spike <= 10))),
                    })
                    timings.append(timing)
                    for target in (0, 1):
                        selected = spike[target_at_position == target]
                        selected_first = first_ticks[target_at_position == target]
                        for local_tick in range(1, 101):
                            count = int(np.sum(selected == local_tick))
                            first_count = int(np.sum(selected_first == local_tick))
                            if count or first_count:
                                timing_histogram.append({"dataset": dataset,
                                                         "cs_end_ms": end, "lambda": rate,
                                                         "target_bit": target,
                                                         "spike_time_ms": local_tick * .1,
                                                         "all_natural_spikes": count,
                                                         "first_natural_spikes": first_count})
    write_csv(ROOT / "summary.csv", summaries)
    write_csv(ROOT / "epoch_accuracy.csv", epochs)
    write_csv(ROOT / "spike_timing_summary.csv", timings)
    write_csv(ROOT / "csv/spike_timing_histogram.csv", timing_histogram)
    return summaries, timings, epochs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--skip-validation", action="store_true")
    args = parser.parse_args()
    if not args.skip_validation:
        baseline_validation()
    if args.validate_only:
        return
    tasks = [(d, e, r) for d in DATASETS for e in ENDS for r in RATES]
    tasks = [t for t in tasks if not (ROOT / "npz" / f"{same_condition_id(*t)}.npz").exists()]
    print(f"Starting {len(tasks)} sweep conditions with {args.workers} workers", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_condition, task): task for task in tasks}
        for future in as_completed(futures):
            task = futures[future]
            future.result()
            aggregate_existing()
    summaries, timings, epochs = aggregate_existing()
    assert len(summaries) == 40 and len(epochs) == 40000
    print("COMPLETE: all 40 conditions, 40000 epoch rows", flush=True)


if __name__ == "__main__":
    main()
