"""W[0,0] 가 학습 중 어느 페어에서 강화되고 어디서 안 되는지 진단.

출력 뉴런 0 이 발화할 때마다 (t_sec, t_pre[0], dt, W[0,0] 전/후)를 로깅한다.
  - 특정 페어에서 로그가 아예 없으면 → 그 페어에서 출력0 발화 실패 (adaptation 등)
  - dt 가 음수면 → 발화 순서 반대(post 먼저) → 약화 분기 → 강화 무효

실행: python diag_hook.py
"""
import sys
from types import SimpleNamespace

import torch

import stdp_t2_inhibition as base

keys = ["device", "output_dir", "run_name", "n_pairs", "initial_weight", "phase1_epochs",
        "sim_duration", "t_cs", "us_delay", "us_duration", "pair_interval", "inter_epoch_gap",
        "i_cs_uA", "i_us_uA", "shunt_amp_uA", "count_threshold", "stdp_curve", "stdp_p", "stdp_d",
        "stdp_ptau", "stdp_dtau", "stdp_sh", "wmin", "wmax", "smoke"]
args = SimpleNamespace(**{k: base.PARAMETER_SET[k] for k in keys})
args.n_pairs = 3

cs, tg = base.load_patterns(3)
dev = torch.device("cpu")
net = base.build_net(args, dev)
net.reset_spike_times()

print("digit pix0:", [float(cs[i][0]) for i in range(3)], " (숫자 0/1/2 의 픽셀0)")
print("letter pix0:", [float(tg[i][0]) for i in range(3)], " (A/B/C 의 픽셀0)")
print("-" * 60)

# 입력 뉴런 0 발화(=STDP-pre 호출) 추적: 언제 t_pre[0] 가 갱신되는지
orig_pre = net._apply_stdp_pre


def hooked_pre(pre_i, t_sec, spec, ctx, gate_fn):
    if pre_i == 0:
        print(f"  [pre0 fired] t={t_sec*1e3:7.3f}ms  -> t_pre[0] set to {t_sec*1e3:.3f}ms")
    orig_pre(pre_i, t_sec, spec, ctx, gate_fn)


net._apply_stdp_pre = hooked_pre

# 출력 뉴런 0 발화 시 W[0,0] 갱신을 후킹해서 로깅
orig_post = net._apply_stdp_post


def hooked_post(post_j, t_sec, spec, ctx, gate_fn):
    if post_j == 0:
        before = float(net.cs_syn.weight.data[0, 0])
        tp0 = float(net.t_pre[0].item())
        orig_post(post_j, t_sec, spec, ctx, gate_fn)
        after = float(net.cs_syn.weight.data[0, 0])
        dt = (t_sec - tp0) * 1e3
        dt_str = f"{dt:+.4f}ms" if tp0 > -1e8 else "pre 미발화"
        print(f"post0 fire @ t={t_sec*1e3:7.3f}ms  dt={dt_str:>12}  "
              f"W[0,0] {before:.3f} -> {after:.3f}  (change {after-before:+.3f})")
    else:
        orig_post(post_j, t_sec, spec, ctx, gate_fn)


net._apply_stdp_post = hooked_post

pw = base.pair_window(args)
onsets = [args.inter_epoch_gap + i * args.pair_interval for i in range(3)]
tot = onsets[-1] + pw
print(f"pair onsets(ms): {[round(o*1e3, 1) for o in onsets]},  US at +{args.us_delay*1e3}ms")
print("-" * 60)

with torch.no_grad():
    net.simulate_training_sequence(
        cs, tg, onsets, spec=base.T2, pair_window=pw, total_duration=tot,
        t_cs=args.t_cs, us_delay=args.us_delay, us_duration=args.us_duration,
        i_cs_uA=args.i_cs_uA, i_us_uA=args.i_us_uA, shunt_amp_uA=args.shunt_amp_uA,
        shunt_post_us=base.PARAMETER_SET["shunt_post_us"], update_weights=True, device=dev)

print("-" * 60)
print("final W[0,0] =", float(net.cs_syn.weight.data[0, 0]))
