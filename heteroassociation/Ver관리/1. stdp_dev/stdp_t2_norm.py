"""뉴런별 가중치 정규화(synaptic scaling)를 넣은 STDP T2 실험.

문제: 곱셈 STDP 는 정규화가 없어 '공통 모드'만 커진다.
  - 약하면 전부 임계 근처(평균 붕괴), 세면 전부 임계 위(포화)
  - target-on / target-off 의 '순서'가 안 갈림 → margin ≈ 0 → exact=0

해결 시도: 매 에폭 끝에 은닉 뉴런별로 들어오는 가중치의 크기를 원래대로 되돌린다.
    w[:, j] <- w[:, j] * (target_norm[j] / ||w[:, j]||)
→ 한 입력이 강해지면 같은 뉴런의 다른 입력이 자동으로 약해짐(경쟁).
→ 레벨(공통 모드)은 고정되고 '차이(차등)'만 남는다.

입력 basis 도 고를 수 있다:
  --basis linear     : 15차원 (원본과 동일, 선형 천장 존재)
  --basis quadratic  : 120차원 (원본 15 + 픽셀쌍 AND 105, 천장 없음)

원본 model.py / stdp_t2_*.py 는 건드리지 않는다.
결과: results/stdp_t2_norm/<run>/

Humans: edit PARAMETER_SET. Agents: use argparse.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

import binary_character as data
from model import HeteroAssociativeSNN, STDPRule
from training_protocols import T2

PARAMETER_SET = {
    "device": "cpu",
    "output_dir": "results/stdp_t2_norm",
    "run_name": None,
    "basis": "quadratic",        # "linear"(15) | "quadratic"(120)
    "norm_mode": "none",           # "l1" | "l2" | "none"(정규화 끄기: 대조군)
    "n_pairs": 10,
    "initial_weight": 15.0,
    "phase1_epochs": 15,
    "sim_duration": 5e-3,
    "t_cs": 0.0,
    "us_delay": 1e-4,
    "us_duration": 100e-6,
    "pair_interval": 6e-3,
    "inter_epoch_gap": 5e-3,
    "shunt_post_us": 1e-3,
    "i_cs_uA": 60.0,
    "i_us_uA": 60.0,
    "shunt_amp_uA": -30.0,
    "count_threshold": 1,
    "stdp_curve": "asym_hebbian",
    "stdp_p": 0.00001,
    "stdp_d": 0.00001,
    "stdp_ptau": 1e-3,
    "stdp_dtau": 5e-4,
    "stdp_sh": 0.0,
    "wmin": 0.0,
    "wmax": 600.0,
    "smoke": False,
    "dt": 1e-6,
    "neuron": {
        "threshold": 1.0,
        "tau_m": 0.0017675010149749302,
        "R_s": 264.02864819141104,
        "adapt_tau": 4.0050604192373275e-05,
        "adapt_g": 0.6111900336984427,
        "adapt_strength": 0.2724982224696975,
        "adapt_reversal": -0.1,
        "adapt_conductance_max": 0.95,
        "adaptation_mode": "conductance",
        "v_reset": 0.0,
        "v_rest": 0.0,
        "input_gain": 0.06529691113673196,
        "input_offset": 0.6065728062013327,
    },
    "synapse": {
        "pdur": 2.911239891501844e-05,
        "reversal_potential": 70.28556104108421,
        "conductance_unit_scale": 3.269842176314188e-07,
        "post_step_voltage": 0.0,
    },
}


def expand_and_features(cs: np.ndarray) -> np.ndarray:
    """(N,15) -> (N,120): 원본 15 + 픽셀쌍 AND(i<j) 105."""
    d = cs.shape[1]
    pairs = np.stack([cs[:, i] * cs[:, j] for i in range(d) for j in range(i + 1, d)], axis=1)
    return np.concatenate([cs, pairs], axis=1).astype(np.float32)


def parse_args():
    p = argparse.ArgumentParser(description="STDP T2 with per-neuron weight normalization.")
    p.add_argument("--device", default=PARAMETER_SET["device"])
    p.add_argument("--output-dir", default=PARAMETER_SET["output_dir"])
    p.add_argument("--run-name", default=PARAMETER_SET["run_name"])
    p.add_argument("--basis", choices=["linear", "quadratic"], default=PARAMETER_SET["basis"])
    p.add_argument("--norm-mode", choices=["l1", "l2", "none"], default=PARAMETER_SET["norm_mode"])
    p.add_argument("--n-pairs", type=int, default=PARAMETER_SET["n_pairs"])
    p.add_argument("--initial-weight", type=float, default=PARAMETER_SET["initial_weight"])
    p.add_argument("--phase1-epochs", type=int, default=PARAMETER_SET["phase1_epochs"])
    p.add_argument("--sim-duration", type=float, default=PARAMETER_SET["sim_duration"])
    p.add_argument("--t-cs", type=float, default=PARAMETER_SET["t_cs"])
    p.add_argument("--us-delay", type=float, default=PARAMETER_SET["us_delay"])
    p.add_argument("--us-duration", type=float, default=PARAMETER_SET["us_duration"])
    p.add_argument("--pair-interval", type=float, default=PARAMETER_SET["pair_interval"])
    p.add_argument("--inter-epoch-gap", type=float, default=PARAMETER_SET["inter_epoch_gap"])
    p.add_argument("--i-cs-uA", type=float, default=PARAMETER_SET["i_cs_uA"])
    p.add_argument("--i-us-uA", type=float, default=PARAMETER_SET["i_us_uA"])
    p.add_argument("--shunt-amp-uA", type=float, default=PARAMETER_SET["shunt_amp_uA"])
    p.add_argument("--count-threshold", type=int, default=PARAMETER_SET["count_threshold"])
    p.add_argument("--stdp-curve", default=PARAMETER_SET["stdp_curve"])
    p.add_argument("--stdp-p", type=float, default=PARAMETER_SET["stdp_p"])
    p.add_argument("--stdp-d", type=float, default=PARAMETER_SET["stdp_d"])
    p.add_argument("--stdp-ptau", type=float, default=PARAMETER_SET["stdp_ptau"])
    p.add_argument("--stdp-dtau", type=float, default=PARAMETER_SET["stdp_dtau"])
    p.add_argument("--stdp-sh", type=float, default=PARAMETER_SET["stdp_sh"])
    p.add_argument("--wmin", type=float, default=PARAMETER_SET["wmin"])
    p.add_argument("--wmax", type=float, default=PARAMETER_SET["wmax"])
    p.add_argument("--smoke", action="store_true", default=PARAMETER_SET["smoke"])
    return p.parse_args()


def load_patterns(n_pairs: int, basis: str):
    digit_images, _ = data.make_digit_dataset()
    letter_images, _ = data.make_letter_dataset()
    cs = digit_images.reshape(10, -1).astype(np.float32)[:n_pairs]
    if basis == "quadratic":
        cs = expand_and_features(cs)
    target = letter_images.reshape(10, -1).astype(np.float32)[:n_pairs]
    return cs, target


def build_net(args, device, n_pre: int):
    dt = float(PARAMETER_SET["dt"])
    neuron_params = {"dt": dt, **PARAMETER_SET["neuron"]}
    synapse_params = {"dt": dt, **PARAMETER_SET["synapse"]}
    stdp = STDPRule.from_curve_name(
        args.stdp_curve, p=args.stdp_p, d=args.stdp_d,
        ptau=args.stdp_ptau, dtau=args.stdp_dtau, sh=args.stdp_sh,
        wmin=args.wmin, wmax=args.wmax,
    )
    return HeteroAssociativeSNN(
        neuron_params=neuron_params,
        synapse_params=synapse_params,
        stdp_learner=stdp,
        n_pre=n_pre,
        n_hidden=15,
        initial_weight=args.initial_weight,
    ).to(device)


def column_norm(w: torch.Tensor, mode: str) -> torch.Tensor:
    """은닉 뉴런(열)별 가중치 크기. (1, n_hidden)"""
    if mode == "l2":
        return w.norm(dim=0, keepdim=True)
    return w.sum(dim=0, keepdim=True)          # l1 (양의 전도도라 합 = L1)


def apply_normalization(net, target_norm: torch.Tensor, mode: str):
    """뉴런별 들어오는 가중치 총량을 target_norm 으로 되돌린다 (경쟁 유도)."""
    if mode == "none":
        return
    w = net.cs_syn.weight.data
    cur = column_norm(w, mode)
    cur = torch.clamp(cur, min=1e-12)
    w.mul_(target_norm / cur)


def pair_window(args) -> float:
    return max(args.us_delay + args.us_duration + PARAMETER_SET["shunt_post_us"], 1e-3)


def train_and_recall(net, cs_all, target_all, args):
    device = next(net.parameters()).device
    net.reset_spike_times()
    n_pairs = cs_all.shape[0]
    pw = pair_window(args)
    spec = T2

    # 정규화 기준: 학습 시작 시점의 뉴런별 가중치 총량을 유지 목표로 삼는다
    target_norm = column_norm(net.cs_syn.weight.data.clone(), args.norm_mode)

    for epoch in range(args.phase1_epochs):
        pre_epoch_shunt = float(args.inter_epoch_gap) if epoch == 0 else 0.0
        base = epoch * (
            ((n_pairs - 1) * args.pair_interval + args.inter_epoch_gap) if n_pairs > 1 else pw
        )
        pair_onsets = [pre_epoch_shunt + base + i * args.pair_interval for i in range(n_pairs)]
        total_duration = pair_onsets[-1] + pw if n_pairs else pw + pre_epoch_shunt
        net.simulate_training_sequence(
            cs_all, target_all, pair_onsets,
            spec=spec, pair_window=pw, total_duration=total_duration,
            t_cs=args.t_cs, us_delay=args.us_delay, us_duration=args.us_duration,
            i_cs_uA=args.i_cs_uA, i_us_uA=args.i_us_uA, shunt_amp_uA=args.shunt_amp_uA,
            shunt_post_us=PARAMETER_SET["shunt_post_us"],
            update_weights=True, device=device,
        )
        apply_normalization(net, target_norm, args.norm_mode)   # ★ 에폭마다 정규화
        net.reset_spike_times()

    pair_rows = []
    for idx in range(n_pairs):
        hidden_spk, _, mem_trace = net.simulate_trial(
            cs_all[idx], us_bits=None, training=False, spec=spec,
            sim_duration=args.sim_duration, t_cs=args.t_cs,
            us_delay=args.us_delay, us_duration=args.us_duration,
            i_cs_uA=args.i_cs_uA, i_us_uA=args.i_us_uA, shunt_amp_uA=args.shunt_amp_uA,
            update_weights=False, record_membrane=True, device=device,
        )
        counts = hidden_spk[0].sum(dim=1).cpu().numpy()
        pred = (counts >= args.count_threshold).astype(np.float32)
        mem_peak = mem_trace.max(axis=1).astype(np.float32)
        tgt = target_all[idx]

        # margin = target-on 막전위 평균 − target-off 막전위 평균 (클수록 잘 갈림)
        on, off = mem_peak[tgt >= 0.5], mem_peak[tgt < 0.5]
        margin = float(on.mean() - off.mean()) if on.size and off.size else 0.0

        pair_rows.append(
            {
                "pair": f"{idx}->{chr(ord('A') + idx)}",
                "exact": bool(np.array_equal(pred, tgt)),
                "bit_acc": float((pred == tgt).mean()),
                "margin": margin,
                "pred_ones": int(pred.sum()),
                "target_ones": int(tgt.sum()),
                "pred": pred.reshape(5, 3).astype(int).tolist(),
                "target": tgt.reshape(5, 3).astype(int).tolist(),
                "mem_peak": mem_peak.reshape(5, 3).tolist(),
            }
        )

    weights = net.get_weights().cpu().numpy()
    bit_accs = [r["bit_acc"] for r in pair_rows]
    margins = [r["margin"] for r in pair_rows]
    return {
        "training": "T2_norm",
        "feature_dim": int(cs_all.shape[1]),
        "neuron_threshold": float(PARAMETER_SET["neuron"]["threshold"]),
        "exact_count": int(sum(r["exact"] for r in pair_rows)),
        "mean_bit_acc": float(np.mean(bit_accs)) if bit_accs else 0.0,
        "mean_margin": float(np.mean(margins)) if margins else 0.0,
        "weight_min": float(weights.min()),
        "weight_max": float(weights.max()),
        "weight_mean": float(weights.mean()),
        "failed_pairs": [r["pair"] for r in pair_rows if not r["exact"]],
        "pair_rows": pair_rows,
        "weights": weights.tolist(),
    }


def main():
    args = parse_args()
    if args.smoke:
        args.n_pairs = min(args.n_pairs, 2)
        args.phase1_epochs = 1
        args.sim_duration = min(args.sim_duration, 2e-3)

    device = torch.device(args.device)
    run_name = args.run_name or time.strftime(f"stdp_t2_norm_{args.basis[:4]}_%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    cs_all, target_all = load_patterns(args.n_pairs, args.basis)
    net = build_net(args, device, n_pre=cs_all.shape[1])
    metrics = train_and_recall(net, cs_all, target_all, args)

    result = {
        "run_name": run_name,
        "paradigm": f"T2 + weight normalization ({args.norm_mode}), basis={args.basis}",
        "config": {
            "basis": args.basis,
            "norm_mode": args.norm_mode,
            "feature_dim": metrics["feature_dim"],
            "n_pairs": args.n_pairs,
            "initial_weight": args.initial_weight,
            "phase1_epochs": args.phase1_epochs,
            "stdp_p": args.stdp_p, "stdp_d": args.stdp_d,
            "shunt_amp_uA": args.shunt_amp_uA,
            "count_threshold": args.count_threshold,
        },
        **{k: v for k, v in metrics.items() if k != "weights"},
    }
    with (out_dir / "metrics.json").open("w") as f:
        json.dump(result, f, indent=2)
    np.save(out_dir / "weights.npy", np.array(metrics["weights"], dtype=np.float32))

    print("=== STDP T2 + weight normalization ===")
    print(
        f"basis={args.basis}({metrics['feature_dim']}d), norm={args.norm_mode}, "
        f"exact={metrics['exact_count']}/{args.n_pairs}, "
        f"bit_acc={metrics['mean_bit_acc']:.3f}, "
        f"margin={metrics['mean_margin']:+.4f}"
    )
    print("per-pair margin:", ", ".join(f"{r['pair']}={r['margin']:+.3f}" for r in metrics["pair_rows"]))
    if metrics["failed_pairs"]:
        print("failed:", ", ".join(metrics["failed_pairs"]))
    print(f"wrote={out_dir}")

    viz = Path(__file__).resolve().parent / "visualization" / "viz_stdp_results.py"
    subprocess.run([sys.executable, str(viz), run_name, "--runs-dir", str(Path(args.output_dir))])


if __name__ == "__main__":
    main()
