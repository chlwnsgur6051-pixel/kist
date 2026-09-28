"""학습 중 스파이크/막전위/가중치 변화 시각화 — 재사용 함수 + standalone.

render_spikes(...) 를 sweep_stdp_t2.py 등에서 import 해 케이스별로 호출한다.
neuron/synapse/STDP 파라미터는 stdp_t2_inhibition.PARAMETER_SET + overrides.

만드는 것 (out_dir):
  spikes_raster.png     입력+출력 스파이크 래스터 (한 평면, 동그라미)
  membrane_weights.png  입력0 스파이크 / 출력0 막전위 / W[src,nid] 시간변화

주의: 학습 dynamics 를 보여주는 illustration 이라 phase1_epochs=1, n_pairs 는 작게(기본 3).

standalone:
  python visualization/viz_spikes.py --n-pairs 3 --neuron 0 --src 0
"""
import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt   # 백엔드는 import 하는 쪽에서 필요시 Agg 설정
import torch

import stdp_t2_inhibition as base

RESULTS = Path(__file__).resolve().parent / "results"

_ARG_KEYS = ["device", "output_dir", "run_name", "n_pairs", "initial_weight", "phase1_epochs",
             "sim_duration", "t_cs", "us_delay", "us_duration", "pair_interval", "inter_epoch_gap",
             "i_cs_uA", "i_us_uA", "shunt_amp_uA", "count_threshold", "stdp_curve", "stdp_p",
             "stdp_d", "stdp_ptau", "stdp_dtau", "stdp_sh", "wmin", "wmax", "smoke"]


def _make_args(overrides):
    ns = SimpleNamespace(**{k: base.PARAMETER_SET[k] for k in _ARG_KEYS})
    for k, v in (overrides or {}).items():
        if k in _ARG_KEYS:
            setattr(ns, k, v)
    return ns


def render_spikes(out_dir, overrides=None, n_pairs=3, neuron=0, src=None,
                  show=False, device="cpu"):
    """학습(record 포함)을 한 번 돌려 스파이크/막전위/가중치 그림을 out_dir 에 저장."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dev = torch.device(device)

    args = _make_args(overrides)
    args.n_pairs = n_pairs
    args.phase1_epochs = 1                     # dynamics illustration

    cs_all, target_all = base.load_patterns(n_pairs)
    net = base.build_net(args, dev)
    net.reset_spike_times()

    pw = base.pair_window(args)
    pair_onsets = [args.inter_epoch_gap + i * args.pair_interval for i in range(n_pairs)]
    total_duration = pair_onsets[-1] + pw

    # 가중치 시간변화 스냅샷: 시냅스 forward 를 감싸 stride 마다 W 저장
    n_steps_est = max(1, int(round(total_duration / float(base.PARAMETER_SET["dt"]))))
    stride = max(1, n_steps_est // 300)
    w_steps, w_snaps = [], []
    _orig_forward = net.cs_syn.forward
    _ctr = {"i": 0}

    def _wrapped(*a, **k):
        out = _orig_forward(*a, **k)
        if _ctr["i"] % stride == 0:
            w_steps.append(_ctr["i"])
            w_snaps.append(net.cs_syn.weight.data.clone().cpu().numpy())
        _ctr["i"] += 1
        return out

    net.cs_syn.forward = _wrapped

    with torch.no_grad():   # autograd 그래프 누적(스택오버플로) 방지
        hidden_spk, pre_spk, meta = net.simulate_training_sequence(
            cs_all, target_all, pair_onsets,
            spec=base.T2, pair_window=pw, total_duration=total_duration,
            t_cs=args.t_cs, us_delay=args.us_delay, us_duration=args.us_duration,
            i_cs_uA=args.i_cs_uA, i_us_uA=args.i_us_uA, shunt_amp_uA=args.shunt_amp_uA,
            shunt_post_us=base.PARAMETER_SET["shunt_post_us"],
            update_weights=True, record_membrane=True, device=dev,
        )
    net.cs_syn.forward = _orig_forward

    dt = float(base.PARAMETER_SET["dt"])
    pre = pre_spk[0].cpu().numpy()
    hid = hidden_spk[0].cpu().numpy()
    mem = meta["hidden_mem"]
    T = pre.shape[1]
    t_ms = np.arange(T) * dt * 1e3
    thr = float(base.PARAMETER_SET["neuron"]["threshold"])

    def shade_events(ax):
        for on in pair_onsets:
            ax.axvline(on * 1e3, color="gray", ls="--", lw=0.8, alpha=0.5)
            us0 = (on + args.us_delay) * 1e3
            us1 = (on + args.us_delay + args.us_duration) * 1e3
            ax.axvspan(us0, us1, color="tab:green", alpha=0.15)

    # ---- Figure 1: 입력+출력 래스터 (동그라미) ----
    fig1, ax = plt.subplots(figsize=(12, 5))
    in_t = [t for i in range(pre.shape[0]) for t in t_ms[pre[i] > 0.5]]
    in_y = [i for i in range(pre.shape[0]) for _ in t_ms[pre[i] > 0.5]]
    out_t = [t for j in range(hid.shape[0]) for t in t_ms[hid[j] > 0.5]]
    out_y = [j for j in range(hid.shape[0]) for _ in t_ms[hid[j] > 0.5]]
    ax.scatter(in_t, in_y, s=40, marker="o", facecolors="none", edgecolors="tab:blue",
               linewidths=1.3)
    ax.scatter(out_t, out_y, s=22, marker="o", color="tab:red")
    shade_events(ax)
    ax.set_ylabel("neuron (0-14)"); ax.set_ylim(-1, 15); ax.set_xlabel("time (ms)")
    ax.set_title(f"Spike raster during training (n_pairs={n_pairs})  blue=input, red=output")
    fig1.tight_layout()
    fig1.savefig(out_dir / "spikes_raster.png", dpi=120, bbox_inches="tight")
    if not show:
        plt.close(fig1)

    # ---- Figure 2: 입력0 스파이크 / 출력nid 막전위 / W[src,nid] ----
    nid = max(0, min(neuron, mem.shape[0] - 1))
    src = nid if src is None else max(0, min(src, mem.shape[0] - 1))
    W_hist = np.array(w_snaps)
    w_t_ms = np.array(w_steps) * dt * 1e3
    w_single = W_hist[:, src, nid]

    fig2, (a0, a1, a2) = plt.subplots(3, 1, figsize=(12, 7.5), sharex=True,
                                      gridspec_kw={"height_ratios": [0.5, 1, 1]})
    ts_in = t_ms[pre[src] > 0.5]
    a0.scatter(ts_in, np.zeros_like(ts_in), s=120, marker="|", color="tab:blue",
               label=f"input neuron {src} spike")
    shade_events(a0)
    a0.set_yticks([]); a0.set_ylabel("input\n(pre)")
    a0.set_title(f"input {src}  ->  output {nid}   (input has no membrane, spikes only)")
    a1.plot(t_ms, mem[nid], lw=0.9, color="tab:purple")
    a1.axhline(thr, color="k", ls=":", lw=1)
    shade_events(a1)
    a1.set_ylabel("output\nmembrane")
    a2.plot(w_t_ms, w_single, lw=1.2, color="tab:orange", marker=".", ms=3)
    shade_events(a2)
    a2.set_ylabel(f"weight\nW[{src},{nid}]"); a2.set_xlabel("time (ms)")
    fig2.tight_layout()
    fig2.savefig(out_dir / "membrane_weights.png", dpi=120, bbox_inches="tight")
    if not show:
        plt.close(fig2)

    return {"input_spikes": int((pre > 0.5).sum()), "output_spikes": int((hid > 0.5).sum())}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-pairs", type=int, default=3)
    p.add_argument("--neuron", type=int, default=0, help="막전위 볼 출력 뉴런 (0~14)")
    p.add_argument("--src", type=int, default=None, help="가중치 볼 입력 뉴런 (기본=출력과 동일)")
    p.add_argument("--device", default="cpu")
    p.add_argument("--no-show", action="store_true")
    cli = p.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    info = render_spikes(RESULTS, overrides=None, n_pairs=cli.n_pairs, neuron=cli.neuron,
                         src=cli.src, show=not cli.no_show, device=cli.device)
    print(f"saved: {RESULTS/'spikes_raster.png'}")
    print(f"saved: {RESULTS/'membrane_weights.png'}")
    print(f"input spikes={info['input_spikes']}, output spikes={info['output_spikes']}")
    if not cli.no_show:
        plt.show()


if __name__ == "__main__":
    main()
