"""입력 정규화(divisive + 부호) 전용 모델. 원본 model.py 는 건드리지 않는다.

원본 StepConductanceSynapse 는 pre 를 bool 로 바꿔버려서(`pre_spike.to(torch.bool)`)
입력의 '크기'와 '부호'가 사라진다. 여기서는 그 값을 그대로 실어

    g_syn(j) = Σ_i x_i · w[i, j]          (x_i 는 실수, 음수 가능)
    I_syn    = g_syn · (E_syn − V)

를 계산하는 GradedStepSynapse 를 쓴다. x_i < 0 이면 g_syn 이 음수가 되어
(E_syn − V) > 0 인 구간에서 자동으로 억제 전류가 된다.

입력 인코딩(런 스크립트에서 생성):
    켜진 픽셀 : +1/k        (k = 활성 픽셀 수)  → 활성부 합 = +1
    꺼진 픽셀 : −1/(D−k)                        → 비활성부 합 = −1
  ⇒ 패턴 밀도와 무관하게 총 구동 규모가 일정(divisive normalization) + 평균 0

가중치 정규화는 넣지 않는다(입력 정규화 효과만 분리해서 보기 위함).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from model import LIFNeuron, StepConductanceSynapse, STDPRule


class GradedStepSynapse(StepConductanceSynapse):
    """pre 입력의 값(부호·크기)을 유지하는 스텝 시냅스."""

    def init(self, batch_size, device, dtype=torch.float32):
        remaining = torch.zeros(batch_size, self.n_pre, device=device, dtype=torch.long)
        amp = torch.zeros(batch_size, self.n_pre, device=device, dtype=dtype)
        if self.delay_steps > 0:
            delay_queue = torch.zeros(
                self.delay_steps, batch_size, self.n_pre, device=device, dtype=dtype
            )
        else:
            delay_queue = None
        return remaining, amp, delay_queue

    def forward(self, pre_value, post_mem, state):
        remaining, amp, delay_queue = state

        if self.delay_steps > 0:
            delayed = delay_queue[0]
            delay_queue = torch.cat([delay_queue[1:], pre_value.unsqueeze(0)], dim=0)
        else:
            delayed = pre_value

        event = delayed != 0                      # 0 이 아닌 값이 들어오면 펄스 시작
        remaining = torch.where(event, torch.full_like(remaining, self.pulse_steps), remaining)
        amp = torch.where(event, delayed.to(amp.dtype), amp)   # 값(부호 포함) 유지

        active = (remaining > 0).to(post_mem.dtype)
        weight = self.weight.to(device=post_mem.device, dtype=post_mem.dtype)
        g_syn = (amp * active).matmul(weight)     # Σ x_i w[i,j]  (음수 가능)
        syn_current = g_syn * (self.reversal_potential - post_mem)

        remaining = torch.clamp(remaining - 1, min=0)
        return syn_current, g_syn, (remaining, amp, delay_queue)


class HeteroAssociativeSNNInorm(nn.Module):
    """정규화된 실수/부호 입력을 받는 15(or 120) -> 15 이형연합 SNN."""

    def __init__(
        self,
        neuron_params: dict,
        synapse_params: dict,
        stdp_learner: STDPRule,
        n_pre: int = 15,
        n_hidden: int = 15,
        initial_weight: float = 50.0,
    ):
        super().__init__()
        self.n_pre = n_pre
        self.n_hidden = n_hidden
        self.dt = float(neuron_params["dt"])
        self.stdp = stdp_learner
        self.input_gain = float(neuron_params["input_gain"])
        self.input_offset = float(neuron_params["input_offset"])
        self.conductance_unit_scale = float(synapse_params.get("conductance_unit_scale", 1.0))

        syn = dict(synapse_params)
        syn["dt"] = self.dt
        syn["weight"] = torch.full((n_pre, n_hidden), float(initial_weight), dtype=torch.float32)
        syn["trainable"] = True
        self.cs_syn = GradedStepSynapse(n_pre, n_hidden, syn)
        self.lif_hidden = LIFNeuron(neuron_params)

        self.register_buffer("t_pre", torch.full((n_pre,), -1e9, dtype=torch.float32))
        self.register_buffer("t_post", torch.full((n_hidden,), -1e9, dtype=torch.float32))

    def reset_spike_times(self):
        self.t_pre.fill_(-1e9)
        self.t_post.fill_(-1e9)

    def step_voltage_from_current_uA(self, current_uA: float) -> float:
        return self.input_gain * float(current_uA) + self.input_offset

    def _stdp_pre(self, pre_i, t_sec, spec, ctx, gate_fn):
        w = self.cs_syn.weight.data
        for post_j in range(self.n_hidden):
            dt_sp = float(self.t_post[post_j].item()) - t_sec
            factor = self.stdp.stdp_factor(dt_sp, float(w[pre_i, post_j].item()))
            eff = gate_fn(factor, spec, post_j, ctx)
            w[pre_i, post_j] = self.stdp.clip_weight(float(w[pre_i, post_j].item()) * eff)

    def _stdp_post(self, post_j, t_sec, spec, ctx, gate_fn):
        w = self.cs_syn.weight.data
        for pre_i in range(self.n_pre):
            dt_sp = t_sec - float(self.t_pre[pre_i].item())
            factor = self.stdp.stdp_factor(dt_sp, float(w[pre_i, post_j].item()))
            eff = gate_fn(factor, spec, post_j, ctx)
            w[pre_i, post_j] = self.stdp.clip_weight(float(w[pre_i, post_j].item()) * eff)

    def _apply_stdp(self, pre, spike, t_sec, spec, ctx, gate_fn):
        # 이진이 아니라 실수 입력이므로 '0 이 아님'으로 pre 이벤트를 판정
        pre_idx = (pre[0] != 0).nonzero(as_tuple=False).view(-1)
        post_idx = (spike[0] > 0.5).nonzero(as_tuple=False).view(-1)
        for i in pre_idx.tolist():
            self.t_pre[i] = t_sec
            self._stdp_pre(i, t_sec, spec, ctx, gate_fn)
        for j in post_idx.tolist():
            self.t_post[j] = t_sec
            self._stdp_post(j, t_sec, spec, ctx, gate_fn)

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
            device = self.cs_syn.weight.device
        cs = torch.as_tensor(cs_bits, device=device, dtype=dtype).view(1, -1)
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
        syn_state = self.cs_syn.init(1, device, dtype)
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
            pre = torch.zeros(1, self.n_pre, device=device, dtype=dtype)
            if step == cs_step:
                pre = cs.clone()

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

            syn_i, _, syn_state = self.cs_syn(pre, mem, syn_state)
            syn_i = syn_i * self.conductance_unit_scale
            spike, mem, adapt = self.lif_hidden(step_v, syn_i, mem, adapt, spike)

            pre_spikes[:, :, step] = pre
            hidden_spikes[:, :, step] = spike
            if mem_trace is not None:
                mem_trace[:, :, step] = mem

            if update_weights:
                self._apply_stdp(pre, spike, t_sec, spec, ctx, effective_stdp_factor)

        if record_membrane:
            return hidden_spikes, pre_spikes, mem_trace[0].detach().cpu().numpy()
        return hidden_spikes, pre_spikes

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
            device = self.cs_syn.weight.device

        cs_all = torch.as_tensor(cs_all, device=device, dtype=dtype)
        us_all = torch.as_tensor(us_all, device=device, dtype=dtype)
        if cs_all.ndim == 1:
            cs_all = cs_all.unsqueeze(0)
            us_all = us_all.unsqueeze(0)
        n_pairs = cs_all.shape[0]
        if len(pair_onsets) != n_pairs:
            raise ValueError(f"pair_onsets length {len(pair_onsets)} != n_pairs {n_pairs}")

        n_steps = max(1, int(round(total_duration / self.dt)))
        contexts, cs_step_to_idx, pair_us_steps = [], {}, []
        for i in range(n_pairs):
            t_cs_abs = float(pair_onsets[i]) + t_cs
            t_us_abs = t_cs_abs + us_delay
            contexts.append(TrialContext(us_mask=us_all[i].detach().cpu().numpy(), t_us=t_us_abs))
            cs_step_to_idx[max(0, int(round(t_cs_abs / self.dt)))] = i
            pair_us_steps.append(us_step_bounds(self.dt, t_us_abs, us_duration, n_steps))

        def active_pair(step: int):
            t_sec = step * self.dt
            for i in range(n_pairs - 1, -1, -1):
                if float(pair_onsets[i]) <= t_sec < float(pair_onsets[i]) + pair_window:
                    return i
            return None

        mem, adapt, spike = self.lif_hidden.init(1, self.n_hidden, device, dtype)
        syn_state = self.cs_syn.init(1, device, dtype)
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
            pre = torch.zeros(1, self.n_pre, device=device, dtype=dtype)
            if step in cs_step_to_idx:
                pre[0] = cs_all[cs_step_to_idx[step]]

            step_v = torch.zeros(1, self.n_hidden, device=device, dtype=dtype)
            for i in range(n_pairs):
                us_step, us_step_end = pair_us_steps[i]
                if us_step <= step < us_step_end:
                    step_v = step_v + us_all[i].view(1, -1) * v_us

            pair_i = active_pair(step)
            in_gap = pair_i is None
            if in_gap:
                us_step, us_step_end, us_mask = 0, 0, None
            else:
                us_step, us_step_end = pair_us_steps[pair_i]
                us_mask = contexts[pair_i].us_mask
            for j in range(self.n_hidden):
                if shunt_active_training_epoch(
                    spec, step, j, us_mask, us_step, us_step_end, in_pair_gap=in_gap,
                ):
                    step_v[0, j] = step_v[0, j] + v_shunt

            syn_i, _, syn_state = self.cs_syn(pre, mem, syn_state)
            syn_i = syn_i * self.conductance_unit_scale
            spike, mem, adapt = self.lif_hidden(step_v, syn_i, mem, adapt, spike)

            pre_spikes[:, :, step] = pre
            hidden_spikes[:, :, step] = spike
            if mem_trace is not None:
                mem_trace[:, :, step] = mem

            if update_weights:
                pair_i = active_pair(step)
                if pair_i is None:
                    continue
                self._apply_stdp(pre, spike, t_sec, spec, contexts[pair_i], effective_stdp_factor)

        meta = {"total_duration": float(total_duration), "n_steps": n_steps}
        if mem_trace is not None:
            meta["hidden_mem"] = mem_trace[0].detach().cpu().numpy()
        return hidden_spikes, pre_spikes, meta

    def get_weights(self) -> torch.Tensor:
        return self.cs_syn.weight.data.clone()
