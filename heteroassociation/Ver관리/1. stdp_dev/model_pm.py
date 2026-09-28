"""±1 (ON/OFF 이중 채널) 입력 인코딩 실험 모델. 원본 model.py 는 건드리지 않는다.

동기: 0/1 입력은 꺼진 픽셀이 아무 기여를 안 해 '밀도 편향/포화'가 생긴다.
±1 로 바꾸면 꺼진 픽셀도 정보를 갖고 '억제'로 작동(매치드 필터). 스파이킹은
음의 스파이크를 못 쏘므로, 망막 ON/OFF 세포처럼 두 채널로 구현한다:

  ON  채널 : 숫자 픽셀=1 이 스파이크 → 흥분성 시냅스 (reversal_potential 높음)
  OFF 채널 : 숫자 픽셀=0 이 스파이크 → 억제성 시냅스 (reversal_potential 낮음)

두 시냅스 모두 양(+)의 전도도 가중치이고 STDP 로 학습된다. 억제는 '음수 가중치'가
아니라 '낮은 reversal_potential' 에서 나온다(I = g*(E_inh - V), V>E_inh 이면 음수).

기본 설계상 선택 (실험 스캐폴드 — 튜닝 필요):
  - ON  시냅스: Hebbian STDP (원본과 동일)
  - OFF 시냅스: anti-Hebbian STDP (억제는 반-헤비안이 자연스러움; 필요시 바꿀 것)
US(교사)/shunt/타이밍 로직은 원본 T2 와 동일.

원본 부품(LIFNeuron, StepConductanceSynapse, STDPRule)을 그대로 재사용한다.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from model import LIFNeuron, StepConductanceSynapse, STDPRule


class HeteroAssociativeSNNPM(nn.Module):
    """digit CS 를 ON/OFF 두 채널로 (±1) 인코딩하는 15->15 이형연합 SNN."""

    def __init__(
        self,
        neuron_params: dict,
        synapse_params: dict,
        stdp_on: STDPRule,
        stdp_off: STDPRule,
        n_pre: int = 15,
        n_hidden: int = 15,
        initial_weight_on: float = 50.0,
        initial_weight_off: float = 50.0,
        inhib_reversal: float = -1.0,
    ):
        super().__init__()
        self.n_pre = n_pre
        self.n_hidden = n_hidden
        self.neuron_params = dict(neuron_params)
        self.dt = float(neuron_params["dt"])
        self.stdp_on = stdp_on
        self.stdp_off = stdp_off
        self.input_gain = float(neuron_params["input_gain"])
        self.input_offset = float(neuron_params["input_offset"])

        scale = float(synapse_params.get("conductance_unit_scale", 1.0))
        self.conductance_unit_scale = scale

        # ON 채널 (흥분성): synapse_params 의 reversal_potential 그대로
        syn_on = dict(synapse_params)
        syn_on["dt"] = self.dt
        syn_on["weight"] = torch.full((n_pre, n_hidden), float(initial_weight_on), dtype=torch.float32)
        syn_on["trainable"] = True
        self.cs_syn_on = StepConductanceSynapse(n_pre, n_hidden, syn_on)

        # OFF 채널 (억제성): reversal_potential 을 낮은 값(inhib_reversal)으로
        syn_off = dict(synapse_params)
        syn_off["dt"] = self.dt
        syn_off["reversal_potential"] = float(inhib_reversal)
        syn_off["weight"] = torch.full((n_pre, n_hidden), float(initial_weight_off), dtype=torch.float32)
        syn_off["trainable"] = True
        self.cs_syn_off = StepConductanceSynapse(n_pre, n_hidden, syn_off)

        self.lif_hidden = LIFNeuron(neuron_params)

        self.register_buffer("t_pre_on", torch.full((n_pre,), -1e9, dtype=torch.float32))
        self.register_buffer("t_pre_off", torch.full((n_pre,), -1e9, dtype=torch.float32))
        self.register_buffer("t_post", torch.full((n_hidden,), -1e9, dtype=torch.float32))

    def reset_spike_times(self):
        self.t_pre_on.fill_(-1e9)
        self.t_pre_off.fill_(-1e9)
        self.t_post.fill_(-1e9)

    def step_voltage_from_current_uA(self, current_uA: float) -> float:
        return self.input_gain * float(current_uA) + self.input_offset

    # --- STDP helpers (시냅스/규칙/t_pre 를 파라미터로 일반화) --------------------
    def _stdp_pre(self, syn, stdp, pre_i, t_sec, spec, ctx, gate_fn):
        w = syn.weight.data
        for post_j in range(self.n_hidden):
            dt_sp = float(self.t_post[post_j].item()) - t_sec
            factor = stdp.stdp_factor(dt_sp, float(w[pre_i, post_j].item()))
            eff = gate_fn(factor, spec, post_j, ctx)
            w[pre_i, post_j] = stdp.clip_weight(float(w[pre_i, post_j].item()) * eff)

    def _stdp_post(self, syn, stdp, t_pre_arr, post_j, t_sec, spec, ctx, gate_fn):
        w = syn.weight.data
        for pre_i in range(self.n_pre):
            dt_sp = t_sec - float(t_pre_arr[pre_i].item())
            factor = stdp.stdp_factor(dt_sp, float(w[pre_i, post_j].item()))
            eff = gate_fn(factor, spec, post_j, ctx)
            w[pre_i, post_j] = stdp.clip_weight(float(w[pre_i, post_j].item()) * eff)

    def _apply_all_stdp(self, pre_on, pre_off, spike, t_sec, spec, ctx, gate_fn):
        pre_on_idx = (pre_on[0] > 0.5).nonzero(as_tuple=False).view(-1)
        pre_off_idx = (pre_off[0] > 0.5).nonzero(as_tuple=False).view(-1)
        post_idx = (spike[0] > 0.5).nonzero(as_tuple=False).view(-1)
        for i in pre_on_idx.tolist():
            self.t_pre_on[i] = t_sec
            self._stdp_pre(self.cs_syn_on, self.stdp_on, i, t_sec, spec, ctx, gate_fn)
        for i in pre_off_idx.tolist():
            self.t_pre_off[i] = t_sec
            self._stdp_pre(self.cs_syn_off, self.stdp_off, i, t_sec, spec, ctx, gate_fn)
        for j in post_idx.tolist():
            self.t_post[j] = t_sec
            self._stdp_post(self.cs_syn_on, self.stdp_on, self.t_pre_on, j, t_sec, spec, ctx, gate_fn)
            self._stdp_post(self.cs_syn_off, self.stdp_off, self.t_pre_off, j, t_sec, spec, ctx, gate_fn)

    def _syn_current(self, pre_on, pre_off, mem, state_on, state_off):
        syn_on, _, state_on = self.cs_syn_on(pre_on, mem, state_on)
        syn_off, _, state_off = self.cs_syn_off(pre_off, mem, state_off)
        syn_i = (syn_on + syn_off) * self.conductance_unit_scale
        return syn_i, state_on, state_off

    # --- 단일 시행 (회상에 사용) ------------------------------------------------
    def simulate_trial(
        self, cs_bits, us_bits=None, *, training, spec, sim_duration,
        t_cs=0.0, us_delay=1e-4, us_duration=100e-6,
        i_cs_uA=60.0, i_us_uA=60.0, shunt_amp_uA=-30.0,
        shunt_pre_pad=5e-6, shunt_post_us=1e-3,
        update_weights=False, record_membrane=False, device=None, dtype=torch.float32,
    ):
        from training_protocols import (
            TrialContext, effective_stdp_factor, shunt_active, us_step_bounds,
        )
        if device is None:
            device = self.cs_syn_on.weight.device
        cs = torch.as_tensor(cs_bits, device=device, dtype=dtype).view(1, -1)
        cs_off = 1.0 - cs                                   # OFF 채널 = 여집합
        us = (
            torch.as_tensor(us_bits, device=device, dtype=dtype).view(1, -1)
            if us_bits is not None
            else torch.zeros(1, self.n_hidden, device=device, dtype=dtype)
        )
        us_mask = us[0].detach().cpu().numpy()
        t_us = t_cs + us_delay
        ctx = TrialContext(us_mask=us_mask, t_us=t_us)

        n_steps = max(1, int(round(sim_duration / self.dt)))
        cs_step = max(0, int(round(t_cs / self.dt)))
        us_step, us_step_end = us_step_bounds(self.dt, t_us, us_duration, n_steps)
        if training and spec.train_shunt:
            shunt_lo_step, shunt_hi_step = 0, n_steps
        else:
            shunt_lo_step, shunt_hi_step = 0, 0

        mem, adapt, spike = self.lif_hidden.init(1, self.n_hidden, device, dtype)
        state_on = self.cs_syn_on.init(1, device, dtype)
        state_off = self.cs_syn_off.init(1, device, dtype)
        hidden_spikes = torch.zeros(1, self.n_hidden, n_steps, device=device, dtype=dtype)
        pre_spikes = torch.zeros(1, self.n_pre, n_steps, device=device, dtype=dtype)
        mem_trace = (
            torch.zeros(1, self.n_hidden, n_steps, device=device, dtype=dtype)
            if record_membrane else None
        )

        v_us = self.step_voltage_from_current_uA(i_us_uA)
        v_shunt = self.step_voltage_from_current_uA(shunt_amp_uA)

        for step in range(n_steps):
            t_sec = step * self.dt
            pre_on = torch.zeros(1, self.n_pre, device=device, dtype=dtype)
            pre_off = torch.zeros(1, self.n_pre, device=device, dtype=dtype)
            if step == cs_step:
                pre_on = cs.clone()
                pre_off = cs_off.clone()

            step_v = torch.zeros(1, self.n_hidden, device=device, dtype=dtype)
            if training and us_step <= step < us_step_end:
                step_v = step_v + us * v_us
            if training:
                for j in range(self.n_hidden):
                    if shunt_active(
                        spec, step, j, us_mask,
                        shunt_lo_step=shunt_lo_step, shunt_hi_step=shunt_hi_step,
                        us_step=us_step, us_step_end=us_step_end,
                    ):
                        step_v[0, j] = step_v[0, j] + v_shunt

            syn_i, state_on, state_off = self._syn_current(pre_on, pre_off, mem, state_on, state_off)
            spike, mem, adapt = self.lif_hidden(step_v, syn_i, mem, adapt, spike)

            pre_spikes[:, :, step] = pre_on
            hidden_spikes[:, :, step] = spike
            if mem_trace is not None:
                mem_trace[:, :, step] = mem

            if update_weights:
                self._apply_all_stdp(pre_on, pre_off, spike, t_sec, spec, ctx, effective_stdp_factor)

        if record_membrane:
            return hidden_spikes, pre_spikes, mem_trace[0].detach().cpu().numpy()
        return hidden_spikes, pre_spikes

    # --- 연속 다중 쌍 학습 -----------------------------------------------------
    def simulate_training_sequence(
        self, cs_all, us_all, pair_onsets, *, spec, pair_window, total_duration,
        t_cs=0.0, us_delay=1e-4, us_duration=100e-6,
        i_cs_uA=60.0, i_us_uA=60.0, shunt_amp_uA=-30.0,
        shunt_pre_pad=5e-6, shunt_post_us=1e-3,
        update_weights=False, record_membrane=False, device=None, dtype=torch.float32,
    ):
        from training_protocols import (
            TrialContext, effective_stdp_factor, shunt_active_training_epoch, us_step_bounds,
        )
        if device is None:
            device = self.cs_syn_on.weight.device

        cs_all = torch.as_tensor(cs_all, device=device, dtype=dtype)
        us_all = torch.as_tensor(us_all, device=device, dtype=dtype)
        if cs_all.ndim == 1:
            cs_all = cs_all.unsqueeze(0)
            us_all = us_all.unsqueeze(0)
        cs_off_all = 1.0 - cs_all
        n_pairs = cs_all.shape[0]
        if len(pair_onsets) != n_pairs:
            raise ValueError(f"pair_onsets length {len(pair_onsets)} != n_pairs {n_pairs}")

        n_steps = max(1, int(round(total_duration / self.dt)))
        contexts = []
        cs_step_to_idx = {}
        pair_us_steps = []
        for i in range(n_pairs):
            t_cs_abs = float(pair_onsets[i]) + t_cs
            t_us_abs = t_cs_abs + us_delay
            us_mask = us_all[i].detach().cpu().numpy()
            contexts.append(TrialContext(us_mask=us_mask, t_us=t_us_abs))
            cs_step = max(0, int(round(t_cs_abs / self.dt)))
            us_step, us_step_end = us_step_bounds(self.dt, t_us_abs, us_duration, n_steps)
            pair_us_steps.append((us_step, us_step_end))
            cs_step_to_idx[cs_step] = i

        def active_pair(step: int):
            t_sec = step * self.dt
            for i in range(n_pairs - 1, -1, -1):
                t_lo = float(pair_onsets[i])
                if t_lo <= t_sec < t_lo + pair_window:
                    return i
            return None

        mem, adapt, spike = self.lif_hidden.init(1, self.n_hidden, device, dtype)
        state_on = self.cs_syn_on.init(1, device, dtype)
        state_off = self.cs_syn_off.init(1, device, dtype)
        hidden_spikes = torch.zeros(1, self.n_hidden, n_steps, device=device, dtype=dtype)
        pre_spikes = torch.zeros(1, self.n_pre, n_steps, device=device, dtype=dtype)
        mem_trace = (
            torch.zeros(1, self.n_hidden, n_steps, device=device, dtype=dtype)
            if record_membrane else None
        )

        v_us = self.step_voltage_from_current_uA(i_us_uA)
        v_shunt = self.step_voltage_from_current_uA(shunt_amp_uA)

        for step in range(n_steps):
            t_sec = step * self.dt
            pre_on = torch.zeros(1, self.n_pre, device=device, dtype=dtype)
            pre_off = torch.zeros(1, self.n_pre, device=device, dtype=dtype)
            if step in cs_step_to_idx:
                idx = cs_step_to_idx[step]
                pre_on[0] = cs_all[idx]
                pre_off[0] = cs_off_all[idx]

            step_v = torch.zeros(1, self.n_hidden, device=device, dtype=dtype)
            for i in range(n_pairs):
                us_step, us_step_end = pair_us_steps[i]
                if us_step <= step < us_step_end:
                    step_v = step_v + us_all[i].view(1, -1) * v_us

            pair_i = active_pair(step)
            in_gap = pair_i is None
            if in_gap:
                us_step, us_step_end = 0, 0
                us_mask = None
            else:
                us_step, us_step_end = pair_us_steps[pair_i]
                us_mask = contexts[pair_i].us_mask
            for j in range(self.n_hidden):
                if shunt_active_training_epoch(
                    spec, step, j, us_mask, us_step, us_step_end, in_pair_gap=in_gap,
                ):
                    step_v[0, j] = step_v[0, j] + v_shunt

            syn_i, state_on, state_off = self._syn_current(pre_on, pre_off, mem, state_on, state_off)
            spike, mem, adapt = self.lif_hidden(step_v, syn_i, mem, adapt, spike)

            pre_spikes[:, :, step] = pre_on
            hidden_spikes[:, :, step] = spike
            if mem_trace is not None:
                mem_trace[:, :, step] = mem

            if update_weights:
                pair_i = active_pair(step)
                if pair_i is None:
                    continue
                ctx = contexts[pair_i]
                self._apply_all_stdp(pre_on, pre_off, spike, t_sec, spec, ctx, effective_stdp_factor)

        meta = {
            "pair_onsets": [float(x) for x in pair_onsets],
            "pair_window": float(pair_window),
            "total_duration": float(total_duration),
            "n_steps": n_steps,
        }
        if mem_trace is not None:
            meta["hidden_mem"] = mem_trace[0].detach().cpu().numpy()
        return hidden_spikes, pre_spikes, meta

    def get_weights(self):
        """(W_on, W_off) 반환."""
        return self.cs_syn_on.weight.data.clone(), self.cs_syn_off.weight.data.clone()
