"""LIF + step-conductance synapse + pairwise STDP for heteroassociation.

Extracted for sharing. No HW-fit (synfit) manifest dependency —
neuron/synapse params are supplied by the caller (see stdp_t2_inhibition.py).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

# [DEBUG] 학습 중 스파이크 감지 / t_pre / t_post 로그 on/off.
# 외부에서 model.LOG_STDP = True 로 켤 수 있다(기본 False).
LOG_STDP = False


class LIFNeuron(nn.Module):
    def __init__(
        self,
        params: dict,
    ):
        """
        Voltage-step + Synaptic-current separated LIF:
            C_m dV/dt = -g_L (V - E_L) + g_step (V_step - V) + I_syn
        """
        super().__init__()
        self.dt = params.get('dt', 1e-3)
        self.threshold = params.get('threshold', 1.0)
        self.v_reset = params.get('v_reset', 0.0)
        self.v_rest = params.get('v_rest', 0.0)

        self.tau_m = params.get('tau_m', 20e-3)
        # self.R_m = params.get('R_m', 1.0)
        self.R_s = params.get('R_s', 1.0)
        self.g_step = (1/self.R_s) if self.R_s > 0 else 0.0
        # step-voltage coupling conductance (g_step = 1 / R_s)
        # self.g_step = params.get('g_step', 0.0)


        self.adapt_tau = params.get('adapt_tau', 10e-3) # refractory period in sample
        self.adapt_g = params.get('adapt_g', 0.1 ) # refractory period in sample
        self.adaptation_mode = params.get('adaptation_mode', 'conditional')
        self.adapt_strength = params.get('adapt_strength', 1.0)
        self.adapt_reversal = params.get('adapt_reversal', 0.0)
        self.adapt_conductance_max = params.get('adapt_conductance_max', 0.95)

        # self.g_step = g_step

        # self.g_tot = self.g_L + self.g_step
        # self.alpha_scalar = math.exp(-dt * self.g_tot / self.C_m)

        # tau_eff = self.C_m / self.g_tot
        # print(f"thr={self.threshold}, dt={self.dt}, tau_eff={tau_eff}, alpha={self.alpha_scalar}")
        self.beta = math.exp(-self.dt / self.tau_m)

    def init(self, batch_size, n_neurons, device, dtype=torch.float32):
        mem = torch.zeros(batch_size, n_neurons, device=device, dtype=dtype)
        adapt = torch.zeros(batch_size, n_neurons, device=device, dtype=dtype)
        spike = torch.zeros(batch_size, n_neurons, device=device, dtype=dtype)
        return mem, adapt, spike

    def forward(self, step_voltage, syn_current, mem, adapt, spike):
        """
        One time-step update

        Args:
            step_voltage: [B, N]   (voltage step input at this timestep)
            syn_current:  [B, N]   (synaptic current at this timestep)
            mem:          [B, N]   (previous membrane voltage)

        Returns:
            spike: [B, N]
            mem:   [B, N]
        """

        # dimensionless coupling strength
        mem = torch.where(spike.bool(),
                          torch.full_like(mem, self.v_reset),
                          mem)
        adapt = math.exp(-self.dt / self.adapt_tau) * adapt
        # adapt = adapt -(adapt/self.adapt_tau)*self.dt

        mem = self.beta * mem + self.g_step * (step_voltage - mem) + syn_current   # synaptic current

        # Adaptation current
        if self.adaptation_mode == 'smooth':
            mem = mem - self.adapt_strength * adapt
        elif self.adaptation_mode == 'conductance':
            conductance = torch.clamp(self.adapt_strength * adapt, min=0.0)
            conductance = torch.minimum(
                conductance,
                torch.full_like(conductance, self.adapt_conductance_max),
            )
            mem = mem - conductance * (mem - self.adapt_reversal)
        else:
            mem = mem - ((mem-adapt)>0).float()*adapt # over zero

        spike = (mem >= self.threshold).to(mem.dtype)

        adapt = adapt + spike*self.adapt_g
        
        # Adaptive Current
        # A = agc, a is decays back to zero with the adaptation (a is zero for low Vm and one for above the spiking threshold) Benda, Maler and Longtin, 2010, J Neurophys


        # # Basic Integration
        # if self.syn == 1: # with Synapse
        #     syn = self.alpha * syn + self.w(input_)
        #     mem = self.beta * mem + syn
        # elif self.syn == 0: #without Synapse
        #     mem = self.beta * mem + self.w(input_) 




        return spike, mem, adapt


class StepConductanceSynapse(nn.Module):
    def __init__(
        self,
        n_pre: int,
        n_post: int,
        params: dict,
    ):
        """
        Step conductance synapse matching the NRN StepCondSyn-style interface:
            pre_spike -> g_syn pulse for pdur
            I_syn = g_syn * (E_syn - V_post)
        """
        super().__init__()
        self.n_pre = n_pre
        self.n_post = n_post
        self.dt = params.get('dt', 1e-3)
        self.pdur = params.get('pdur', params.get('step_duration', 1e-3))
        self.delay = params.get('delay', 0.0)
        self.reversal_potential = params.get(
            'reversal_potential',
            params.get('e_syn', 1.0),
        )
        self.pulse_steps = max(1, int(round(self.pdur / self.dt)))
        self.delay_steps = max(0, int(round(self.delay / self.dt)))

        weight = params.get('weight', 1.0)
        trainable = params.get('trainable', False)
        if torch.is_tensor(weight):
            weight_tensor = weight.detach().clone().float()
        else:
            weight_tensor = torch.full((n_pre, n_post), float(weight), dtype=torch.float32)
        if weight_tensor.ndim == 0:
            weight_tensor = torch.full((n_pre, n_post), float(weight_tensor.item()))
        elif weight_tensor.shape != (n_pre, n_post):
            raise ValueError(f"weight must have shape {(n_pre, n_post)}, got {tuple(weight_tensor.shape)}")

        if trainable:
            self.weight = nn.Parameter(weight_tensor)
        else:
            self.register_buffer('weight', weight_tensor)

    def init(self, batch_size, device, dtype=torch.float32):
        remaining = torch.zeros(batch_size, self.n_pre, device=device, dtype=torch.long)
        if self.delay_steps > 0:
            delay_queue = torch.zeros(
                self.delay_steps,
                batch_size,
                self.n_pre,
                device=device,
                dtype=torch.bool,
            )
        else:
            delay_queue = None
        return remaining, delay_queue

    def forward(self, pre_spike, post_mem, state):
        remaining, delay_queue = state
        pre_event = pre_spike.to(torch.bool)

        if self.delay_steps > 0:
            delayed_event = delay_queue[0]
            delay_queue = torch.cat([delay_queue[1:], pre_event.unsqueeze(0)], dim=0)
        else:
            delayed_event = pre_event

        pulse_steps = torch.full_like(remaining, self.pulse_steps)
        remaining = torch.where(delayed_event, pulse_steps, remaining)
        active = remaining > 0

        weight = self.weight.to(device=post_mem.device, dtype=post_mem.dtype)
        g_syn = active.to(post_mem.dtype).matmul(weight)
        syn_current = g_syn * (self.reversal_potential - post_mem)

        remaining = torch.clamp(remaining - 1, min=0)
        return syn_current, g_syn, (remaining, delay_queue)


STDP_CURVE_TYPES = {
    "asym_hebbian": 0,
    "sym_hebbian": 1,
    "asym_anti_hebbian": 2,
    "sym_anti_hebbian": 3,
    "mexican_hat": 4,
    "shifted_mexican_hat": 5,
}


class STDPRule:
    """Port of ``neuron/mechanisms/stepcondsyn_stdp.mod`` stdp_factor (times in seconds)."""

    def __init__(
        self,
        p=1.0,
        d=0.01,
        ptau=1e-3,
        dtau=5e-4,
        wmin=0.0,
        wmax=1140.0,
        curve_type=0,
        sh=0.0,
        is_hebb=True,
    ):
        self.p = float(p)
        self.d = float(d)
        self.ptau = float(ptau)
        self.dtau = float(dtau)
        self.wmin = float(wmin)
        self.wmax = float(wmax)
        self.curve_type = int(curve_type)
        self.sh = float(sh)
        self.is_hebb = bool(is_hebb)

    @classmethod
    def from_curve_name(cls, name: str, **kwargs):
        key = name.strip().lower().replace("-", "_")
        if key not in STDP_CURVE_TYPES:
            raise ValueError(f"Unknown STDP curve {name!r}")
        return cls(curve_type=STDP_CURVE_TYPES[key], **kwargs)

    def stdp_factor(self, dt_seconds: float, weight: float) -> float:
        dt_eff = float(dt_seconds) + self.sh
        nw = float(weight)
        ct = self.curve_type
        eps = 1e-9

        if abs(dt_eff) < eps:
            if ct == 1:
                return 1.0 + self.p * abs(self.wmax - nw)
            if ct == 3:
                return 1.0 - self.d * abs(nw - self.wmin)
            if ct in (4, 5):
                win = self.p - self.d
                if win > 0:
                    return 1.0 + win * abs(self.wmax - nw)
                return 1.0 + win * abs(nw - self.wmin)
            return 1.0

        if ct == 1:
            ex = -abs(dt_eff) / self.ptau
            ex = max(ex, -700.0)
            return 1.0 + self.p * abs(self.wmax - nw) * math.exp(ex)
        if ct == 3:
            ex = -abs(dt_eff) / self.dtau
            ex = max(ex, -700.0)
            return 1.0 - self.d * abs(nw - self.wmin) * math.exp(ex)
        if ct in (4, 5):
            if self.ptau <= 0 or self.dtau <= 0:
                return 1.0
            x = dt_eff / self.ptau
            ex = max(-0.5 * x * x, -700.0)
            win = self.p * math.exp(ex)
            x = dt_eff / self.dtau
            ex = max(-0.5 * x * x, -700.0)
            win = win - self.d * math.exp(ex)
            if win > 0:
                return 1.0 + win * abs(self.wmax - nw)
            return 1.0 + win * abs(nw - self.wmin)
        if ct == 2 or (ct == 0 and not self.is_hebb):
            if dt_eff < 0:
                ex = -abs(dt_eff) / self.ptau
                ex = max(ex, -700.0)
                return 1.0 + self.p * abs(self.wmax - nw) * math.exp(ex)
            ex = -abs(dt_eff) / self.dtau
            ex = max(ex, -700.0)
            return 1.0 - self.d * abs(nw - self.wmin) * math.exp(ex)

        if dt_eff > 0:
            ex = -abs(dt_eff) / self.ptau
            ex = max(ex, -700.0)
            return 1.0 + self.p * abs(self.wmax - nw) * math.exp(ex)
        ex = -abs(dt_eff) / self.dtau
        ex = max(ex, -700.0)
        return 1.0 - self.d * abs(nw - self.wmin) * math.exp(ex)

    def clip_weight(self, weight: float) -> float:
        return float(min(max(weight, self.wmin), self.wmax))


class HeteroAssociativeSNN(nn.Module):
    """
    Digit CS (15 pre) -> hidden (15 LIF) via StepConductanceSynapse + STDP.
    US teacher: sustained step_voltage on US-on hidden (training only).
    T2 shunt: negative step on US-off hidden during training.
    """

    def __init__(
        self,
        neuron_params: dict,
        synapse_params: dict,
        stdp_learner: STDPRule,
        n_pre: int = 15,
        n_hidden: int = 15,
        initial_weight: float = 60.0,
    ):
        super().__init__()
        self.n_pre = n_pre
        self.n_hidden = n_hidden
        self.neuron_params = dict(neuron_params)
        self.dt = float(neuron_params["dt"])
        self.stdp = stdp_learner
        self.input_gain = float(neuron_params["input_gain"])
        self.input_offset = float(neuron_params["input_offset"])

        scale = float(synapse_params.get("conductance_unit_scale", 1.0))
        syn_params = dict(synapse_params)
        syn_params["dt"] = self.dt
        syn_params["weight"] = torch.full(
            (n_pre, n_hidden),
            float(initial_weight),
            dtype=torch.float32,
        )
        syn_params["trainable"] = True
        self.cs_syn = StepConductanceSynapse(n_pre, n_hidden, syn_params)
        self.conductance_unit_scale = scale
        self.lif_hidden = LIFNeuron(neuron_params)

        self.register_buffer(
            "t_pre",
            torch.full((n_pre,), -1e9, dtype=torch.float32),
        )
        self.register_buffer(
            "t_post",
            torch.full((n_hidden,), -1e9, dtype=torch.float32),
        )

    def reset_spike_times(self):
        self.t_pre.fill_(-1e9)
        self.t_post.fill_(-1e9)

    def step_voltage_from_current_uA(self, current_uA: float) -> float:
        return self.input_gain * float(current_uA) + self.input_offset

    def _apply_stdp_pre(self, pre_i: int, t_sec: float, spec, ctx, gate_fn) -> None:
        w = self.cs_syn.weight.data
        for post_j in range(self.n_hidden):
            dt_sp = float(self.t_post[post_j].item()) - t_sec
            factor = self.stdp.stdp_factor(dt_sp, float(w[pre_i, post_j].item()))
            eff = gate_fn(factor, spec, post_j, ctx)
            w[pre_i, post_j] = self.stdp.clip_weight(float(w[pre_i, post_j].item()) * eff)

    def _apply_stdp_post(self, post_j: int, t_sec: float, spec, ctx, gate_fn) -> None:
        w = self.cs_syn.weight.data
        for pre_i in range(self.n_pre):
            dt_sp = t_sec - float(self.t_pre[pre_i].item())
            factor = self.stdp.stdp_factor(dt_sp, float(w[pre_i, post_j].item()))
            eff = gate_fn(factor, spec, post_j, ctx)
            w[pre_i, post_j] = self.stdp.clip_weight(float(w[pre_i, post_j].item()) * eff)

    def simulate_trial(
        self,
        cs_bits,
        us_bits=None,
        *,
        training: bool,
        spec,
        sim_duration: float,
        t_cs: float = 0.0,
        us_delay: float = 1e-4,
        us_duration: float = 100e-6,
        i_cs_uA: float = 60.0,
        i_us_uA: float = 60.0,
        shunt_amp_uA: float = -30.0,
        shunt_pre_pad: float = 5e-6,
        shunt_post_us: float = 1e-3,
        update_weights: bool = False,
        record_membrane: bool = False,
        device=None,
        dtype=torch.float32,
    ):
        from training_protocols import (
            TrialContext,
            effective_stdp_factor,
            shunt_active,
            us_step_bounds,
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
            if record_membrane
            else None
        )

        v_cs = self.step_voltage_from_current_uA(i_cs_uA)
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
                        spec,
                        step,
                        j,
                        us_mask,
                        shunt_lo_step=shunt_lo_step,
                        shunt_hi_step=shunt_hi_step,
                        us_step=us_step,
                        us_step_end=us_step_end,
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
                pre_idx = (pre[0] > 0.5).nonzero(as_tuple=False).view(-1)
                post_idx = (spike[0] > 0.5).nonzero(as_tuple=False).view(-1)
                for i in pre_idx.tolist():
                    self.t_pre[i] = t_sec
                    self._apply_stdp_pre(i, t_sec, spec, ctx, effective_stdp_factor)
                for j in post_idx.tolist():
                    self.t_post[j] = t_sec
                    self._apply_stdp_post(j, t_sec, spec, ctx, effective_stdp_factor)

        if record_membrane:
            return hidden_spikes, pre_spikes, mem_trace[0].detach().cpu().numpy()
        return hidden_spikes, pre_spikes

    def simulate_training_sequence(
        self,
        cs_all,
        us_all,
        pair_onsets,
        *,
        spec,
        pair_window: float,
        total_duration: float,
        t_cs: float = 0.0,
        us_delay: float = 1e-4,
        us_duration: float = 100e-6,
        i_cs_uA: float = 60.0,
        i_us_uA: float = 60.0,
        shunt_amp_uA: float = -30.0,
        shunt_pre_pad: float = 5e-6,
        shunt_post_us: float = 1e-3,
        update_weights: bool = False,
        record_membrane: bool = False,
        device=None,
        dtype=torch.float32,
    ):
        """Continuous multi-pair training: one timeline, membrane/synaptic state carried across pairs."""
        from training_protocols import (
            TrialContext,
            effective_stdp_factor,
            shunt_active_training_epoch,
            us_step_bounds,
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
        contexts = []
        cs_step_to_idx = {}
        us_ranges = []
        pair_cs_steps = []
        pair_us_steps = []

        for i in range(n_pairs):
            t_cs_abs = float(pair_onsets[i]) + t_cs
            t_us_abs = t_cs_abs + us_delay
            us_mask = us_all[i].detach().cpu().numpy()
            contexts.append(TrialContext(us_mask=us_mask, t_us=t_us_abs))

            cs_step = max(0, int(round(t_cs_abs / self.dt)))
            us_step, us_step_end = us_step_bounds(self.dt, t_us_abs, us_duration, n_steps)
            pair_cs_steps.append(cs_step)
            pair_us_steps.append((us_step, us_step_end))
            cs_step_to_idx[cs_step] = i
            us_ranges.append((us_step, us_step_end, us_all[i]))

        def active_pair(step: int):
            t_sec = step * self.dt
            eps = self.dt * 0.5
            for i in range(n_pairs - 1, -1, -1):
                t_lo = float(pair_onsets[i])
                if t_lo - eps <= t_sec < t_lo + pair_window:
                    return i
            return None

        mem, adapt, spike = self.lif_hidden.init(1, self.n_hidden, device, dtype)
        syn_state = self.cs_syn.init(1, device, dtype)
        hidden_spikes = torch.zeros(1, self.n_hidden, n_steps, device=device, dtype=dtype)
        pre_spikes = torch.zeros(1, self.n_pre, n_steps, device=device, dtype=dtype)
        mem_trace = (
            torch.zeros(1, self.n_hidden, n_steps, device=device, dtype=dtype)
            if record_membrane
            else None
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
                us_step, us_step_end = 0, 0
                us_mask = None
            else:
                us_step, us_step_end = pair_us_steps[pair_i]
                us_mask = contexts[pair_i].us_mask
            for j in range(self.n_hidden):
                if shunt_active_training_epoch(
                    spec,
                    step,
                    j,
                    us_mask,
                    us_step,
                    us_step_end,
                    in_pair_gap=in_gap,
                ):
                    step_v[0, j] = step_v[0, j] + v_shunt

            syn_i, _, syn_state = self.cs_syn(pre, mem, syn_state)
            syn_i = syn_i * self.conductance_unit_scale
            spike, mem, adapt = self.lif_hidden(step_v, syn_i, mem, adapt, spike)

            pre_spikes[:, :, step] = pre
            hidden_spikes[:, :, step] = spike
            if mem_trace is not None:
                mem_trace[:, :, step] = mem

            if LOG_STDP and update_weights:
                # 스파이크 감지 여부 + active_pair 게이트 결과. gate=None 이면
                # 스파이크가 있어도 아래 STDP 블록이 통째로 스킵된다(버그 흔적).
                pre_f = (pre[0] > 0.5).nonzero(as_tuple=False).view(-1).tolist()
                post_f = (spike[0] > 0.5).nonzero(as_tuple=False).view(-1).tolist()
                if pre_f or post_f:
                    pi = active_pair(step)
                    skip = "  <<< SKIP (active_pair=None -> STDP 스킵)" if pi is None else ""
                    print(f"[spk]   step={step:>7} t_sec={t_sec!r}  "
                          f"pre_fired={pre_f}  post_fired={post_f}  active_pair={pi}{skip}")
                    # t_lo <= t_sec < t_lo + pair_window 의 실제 값과 판정 (페어별)
                    for i, o in enumerate(pair_onsets):
                        t_lo = float(o); t_hi = t_lo + pair_window
                        inside = (t_lo <= t_sec < t_hi)
                        print(f"[bound]   pair{i}: t_lo={t_lo!r}  t_sec={t_sec!r}  "
                              f"t_lo+pw={t_hi!r}  ->  (t_lo<=t_sec<t_lo+pw)={inside}")

            if update_weights:
                pair_i = active_pair(step)
                if pair_i is None:
                    continue
                ctx = contexts[pair_i]
                pre_idx = (pre[0] > 0.5).nonzero(as_tuple=False).view(-1)
                post_idx = (spike[0] > 0.5).nonzero(as_tuple=False).view(-1)
                for i in pre_idx.tolist():
                    self.t_pre[i] = t_sec
                    self._apply_stdp_pre(i, t_sec, spec, ctx, effective_stdp_factor)
                for j in post_idx.tolist():
                    self.t_post[j] = t_sec
                    self._apply_stdp_post(j, t_sec, spec, ctx, effective_stdp_factor)

        meta = {
            "pair_onsets": [float(x) for x in pair_onsets],
            "pair_window": float(pair_window),
            "pair_cs_steps": pair_cs_steps,
            "total_duration": float(total_duration),
            "n_steps": n_steps,
        }
        if mem_trace is not None:
            meta["hidden_mem"] = mem_trace[0].detach().cpu().numpy()
        return hidden_spikes, pre_spikes, meta

    def load_weights(self, weight: torch.Tensor):
        self.cs_syn.weight.data.copy_(weight)

    def get_weights(self) -> torch.Tensor:
        return self.cs_syn.weight.data.clone()


