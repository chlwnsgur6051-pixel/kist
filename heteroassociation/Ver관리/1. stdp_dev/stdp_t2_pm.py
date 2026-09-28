"""±1 (ON/OFF 이중 채널) 입력 인코딩 실험 실행 스크립트.

원본 stdp_t2_inhibition.py 와 동일한 T2 학습/회상 파이프라인이되,
입력을 HeteroAssociativeSNNPM 로 ON/OFF ±1 인코딩한다(model_pm.py).
원본 파일들은 건드리지 않는다.

새 파라미터:
  --inhib-reversal      OFF(억제) 채널 reversal_potential (낮을수록 강한 억제)
  --initial-weight-off  OFF 채널 초기 전도도
  --off-curve           OFF 채널 STDP 곡선 (기본 asym_anti_hebbian)
  --off-stdp-p/-d       OFF 채널 STDP 학습률

결과: results/stdp_t2_pm/<run>/ 에 metrics.json, weights.npy(W_on), weights_off.npy
자동으로 visualization/viz_stdp_results.py 호출.

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
from model import STDPRule
from model_pm import HeteroAssociativeSNNPM
from training_protocols import T2

PARAMETER_SET = {
    "device": "cpu",
    "output_dir": "results/stdp_t2_pm",
    "run_name": None,
    "n_pairs": 10,
    "initial_weight": 50.0,       # ON 채널 초기 가중치
    "initial_weight_off": 50.0,   # OFF 채널 초기 가중치
    # OFF 채널 reversal (억제). 흥분성(+70)과 대칭이 되도록 -70 근처로.
    # -1 처럼 0에 가까우면 구동력이 작아 억제가 거의 무력해진다.
    "inhib_reversal": -70.0,
    "phase1_epochs": 1,
    # Timing (seconds)
    "sim_duration": 5e-3,
    "t_cs": 0.0,
    "us_delay": 1e-4,
    "us_duration": 100e-6,
    "pair_interval": 6e-3,
    "inter_epoch_gap": 5e-3,
    "shunt_post_us": 1e-3,
    # Currents (uA)
    "i_cs_uA": 60.0,
    "i_us_uA": 60.0,
    "shunt_amp_uA": -30.0,
    "count_threshold": 1,
    # STDP (ON = Hebbian, OFF = anti-Hebbian)
    "stdp_curve": "asym_hebbian",
    "stdp_p": 0.001,
    "stdp_d": 0.01,
    "off_curve": "asym_anti_hebbian",
    "off_stdp_p": 0.001,
    "off_stdp_d": 0.01,
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


def parse_args():
    p = argparse.ArgumentParser(description="STDP T2 with +/-1 ON/OFF input encoding.")
    p.add_argument("--device", default=PARAMETER_SET["device"])
    p.add_argument("--output-dir", default=PARAMETER_SET["output_dir"])
    p.add_argument("--run-name", default=PARAMETER_SET["run_name"])
    p.add_argument("--n-pairs", type=int, default=PARAMETER_SET["n_pairs"])
    p.add_argument("--initial-weight", type=float, default=PARAMETER_SET["initial_weight"])
    p.add_argument("--initial-weight-off", type=float, default=PARAMETER_SET["initial_weight_off"])
    p.add_argument("--inhib-reversal", type=float, default=PARAMETER_SET["inhib_reversal"])
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
    p.add_argument("--off-curve", default=PARAMETER_SET["off_curve"])
    p.add_argument("--off-stdp-p", type=float, default=PARAMETER_SET["off_stdp_p"])
    p.add_argument("--off-stdp-d", type=float, default=PARAMETER_SET["off_stdp_d"])
    p.add_argument("--stdp-ptau", type=float, default=PARAMETER_SET["stdp_ptau"])
    p.add_argument("--stdp-dtau", type=float, default=PARAMETER_SET["stdp_dtau"])
    p.add_argument("--stdp-sh", type=float, default=PARAMETER_SET["stdp_sh"])
    p.add_argument("--wmin", type=float, default=PARAMETER_SET["wmin"])
    p.add_argument("--wmax", type=float, default=PARAMETER_SET["wmax"])
    p.add_argument("--smoke", action="store_true", default=PARAMETER_SET["smoke"])
    return p.parse_args()


def load_patterns(n_pairs: int):
    digit_images, _ = data.make_digit_dataset()
    letter_images, _ = data.make_letter_dataset()
    cs = digit_images.reshape(10, -1).astype(np.float32)[:n_pairs]
    target = letter_images.reshape(10, -1).astype(np.float32)[:n_pairs]
    return cs, target


def build_net(args, device):
    dt = float(PARAMETER_SET["dt"])
    neuron_params = {"dt": dt, **PARAMETER_SET["neuron"]}
    synapse_params = {"dt": dt, **PARAMETER_SET["synapse"]}
    stdp_on = STDPRule.from_curve_name(
        args.stdp_curve, p=args.stdp_p, d=args.stdp_d,
        ptau=args.stdp_ptau, dtau=args.stdp_dtau, sh=args.stdp_sh,
        wmin=args.wmin, wmax=args.wmax,
    )
    stdp_off = STDPRule.from_curve_name(
        args.off_curve, p=args.off_stdp_p, d=args.off_stdp_d,
        ptau=args.stdp_ptau, dtau=args.stdp_dtau, sh=args.stdp_sh,
        wmin=args.wmin, wmax=args.wmax,
    )
    return HeteroAssociativeSNNPM(
        neuron_params=neuron_params,
        synapse_params=synapse_params,
        stdp_on=stdp_on,
        stdp_off=stdp_off,
        initial_weight_on=args.initial_weight,
        initial_weight_off=args.initial_weight_off,
        inhib_reversal=args.inhib_reversal,
    ).to(device)


def pair_window(args) -> float:
    return max(args.us_delay + args.us_duration + PARAMETER_SET["shunt_post_us"], 1e-3)


def train_and_recall(net, cs_all, target_all, args):
    device = next(net.parameters()).device
    net.reset_spike_times()
    n_pairs = cs_all.shape[0]
    pw = pair_window(args)
    spec = T2

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
        pair_rows.append(
            {
                "pair": f"{idx}->{chr(ord('A') + idx)}",
                "exact": bool(np.array_equal(pred, tgt)),
                "bit_acc": float((pred == tgt).mean()),
                "pred_ones": int(pred.sum()),
                "target_ones": int(tgt.sum()),
                "pred": pred.reshape(5, 3).astype(int).tolist(),
                "target": tgt.reshape(5, 3).astype(int).tolist(),
                "mem_peak": mem_peak.reshape(5, 3).tolist(),
            }
        )

    w_on, w_off = net.get_weights()
    w_on = w_on.cpu().numpy()
    w_off = w_off.cpu().numpy()
    bit_accs = [r["bit_acc"] for r in pair_rows]
    return {
        "training": "T2_pm_onoff",
        "neuron_threshold": float(PARAMETER_SET["neuron"]["threshold"]),
        "exact_count": int(sum(r["exact"] for r in pair_rows)),
        "mean_bit_acc": float(np.mean(bit_accs)) if bit_accs else 0.0,
        "weight_on_min": float(w_on.min()), "weight_on_max": float(w_on.max()),
        "weight_off_min": float(w_off.min()), "weight_off_max": float(w_off.max()),
        "failed_pairs": [r["pair"] for r in pair_rows if not r["exact"]],
        "pair_rows": pair_rows,
        "weights_on": w_on.tolist(),
        "weights_off": w_off.tolist(),
    }


def main():
    args = parse_args()
    if args.smoke:
        args.n_pairs = min(args.n_pairs, 2)
        args.phase1_epochs = 1
        args.sim_duration = min(args.sim_duration, 2e-3)

    device = torch.device(args.device)
    run_name = args.run_name or time.strftime("stdp_t2_pm_%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    cs_all, target_all = load_patterns(args.n_pairs)
    net = build_net(args, device)
    metrics = train_and_recall(net, cs_all, target_all, args)

    result = {
        "run_name": run_name,
        "paradigm": "T2 +/-1 ON/OFF encoding",
        "config": {
            "n_pairs": args.n_pairs,
            "initial_weight": args.initial_weight,
            "initial_weight_off": args.initial_weight_off,
            "inhib_reversal": args.inhib_reversal,
            "phase1_epochs": args.phase1_epochs,
            "stdp_curve": args.stdp_curve, "off_curve": args.off_curve,
            "stdp_p": args.stdp_p, "off_stdp_p": args.off_stdp_p,
            "shunt_amp_uA": args.shunt_amp_uA,
        },
        **{k: v for k, v in metrics.items() if k not in ("weights_on", "weights_off")},
    }
    with (out_dir / "metrics.json").open("w") as f:
        json.dump(result, f, indent=2)
    np.save(out_dir / "weights.npy", np.array(metrics["weights_on"], dtype=np.float32))       # ON
    np.save(out_dir / "weights_off.npy", np.array(metrics["weights_off"], dtype=np.float32))   # OFF

    print("=== STDP T2 (+/-1 ON/OFF encoding) ===")
    print(
        f"exact={metrics['exact_count']}/{args.n_pairs}, "
        f"bit_acc={metrics['mean_bit_acc']:.3f}, "
        f"W_on=[{metrics['weight_on_min']:.1f},{metrics['weight_on_max']:.1f}] "
        f"W_off=[{metrics['weight_off_min']:.1f},{metrics['weight_off_max']:.1f}]"
    )
    if metrics["failed_pairs"]:
        print("failed:", ", ".join(metrics["failed_pairs"]))
    print(f"wrote={out_dir}")

    # 결과 시각화 자동 호출 (viz 는 weights.npy=W_on + metrics.json 사용)
    viz = Path(__file__).resolve().parent / "visualization" / "viz_stdp_results.py"
    subprocess.run([sys.executable, str(viz), run_name, "--runs-dir", str(Path(args.output_dir))])


if __name__ == "__main__":
    main()
