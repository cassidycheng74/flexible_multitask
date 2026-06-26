"""
Prototype task implementations for new task ideas.

These are draft tasks and are not yet integrated into task.py.
T20 here is a supervised proxy for CueResponseAssoc that fits the
existing Driscoll input/output format.
"""

# T20 Integration later:
# 1. copy cueassoc into task.py
# 2. add 'cueassoc' to rules_dict, probably under 'new_tasks_small'
# 3. add 'cueassoc': cueassoc to rule_mapping
# 4. add 'cueassoc': 'CueAssoc' to rule_name

from __future__ import division
import numpy as np

from task import Trial

# Scalar channel indices for prototype use.
# When integrating into task.py, replace with config-derived values.
SCALAR_IN_A = 5  # real-valued input channel A
SCALAR_IN_B = 6  # real-valued input channel B
SCALAR_OUT  = 1  # scalar response written to y[:, :, 1]

def pulsecounting(config, mode, **kwargs):
    """
    T13: PulseCounting.
 
    Watch a train of N irregular pulses on the scalar input channel.
    After a delay, output N / N_max as a scalar on the response channel.
 
    Requires discrete state advancement — not solvable by simple integration.
 
    Inputs:  fixation (ch 0), pulses on SCALAR_IN_A (ch 5)
    Outputs: fixation (ch 0), scalar count on SCALAR_OUT (ch 1)
    """
    dt = config['dt']
    rng = config['rng']
    n_max = 6  # maximum pulse count; N drawn from {2, ..., n_max}
 
    if mode == 'random':
        batch_size = kwargs['batch_size']
        n_pulses = rng.randint(2, n_max + 1, size=batch_size)
 
        pulse_ons  = int(rng.uniform(300, 700) / dt)
        pulse_dur  = int(rng.uniform(1500, 2500) / dt)
        delay_dur  = int(rng.uniform(300, 700) / dt)
        fix_offs   = pulse_ons + pulse_dur + delay_dur
        tdim       = fix_offs + int(rng.uniform(300, 700) / dt)
 
    elif mode == 'test':
        batch_size = n_max - 1  # one trial per count: N = 2..6
        n_pulses   = np.arange(2, n_max + 1)
        pulse_ons  = int(500 / dt)
        pulse_dur  = int(2000 / dt)
        delay_dur  = int(500 / dt)
        fix_offs   = pulse_ons + pulse_dur + delay_dur
        tdim       = fix_offs + int(500 / dt)
 
    else:
        raise ValueError('Unknown mode: ' + str(mode))
 
    trial = Trial(config, tdim, batch_size)
    trial.add('fix_in', offs=fix_offs)
    trial.add('fix_out', offs=fix_offs)
 
    pulse_end = pulse_ons + pulse_dur
    for i in range(batch_size):
        # Place N pulses at random non-overlapping times within the window.
        available = np.arange(pulse_ons, pulse_end - 3)
        positions = sorted(rng.choice(available, size=n_pulses[i], replace=False))
        for p in positions:
            trial.x[p:p + 3, i, SCALAR_IN_A] = 1.0
 
    # Scalar count output: N / N_max, held for the whole response epoch.
    for i in range(batch_size):
        trial.y[fix_offs:, i, SCALAR_OUT] = n_pulses[i] / n_max
 
    check_ons = fix_offs + int(100 / dt)
    trial.add_c_mask(pre_offs=fix_offs, post_ons=check_ons)
 
    trial.epochs = {
        'fix1'   : (None, pulse_ons),
        'pulses' : (pulse_ons, pulse_ons + pulse_dur),
        'delay1' : (pulse_ons + pulse_dur, fix_offs),
        'go1'    : (fix_offs, None),
    }
 
    # Metadata for debugging.
    trial.n_pulses = n_pulses
    return trial
 
 
# ---------------------------------------------------------------------------
# T14: IntervalReproduction
# ---------------------------------------------------------------------------
 
def intervalreproduction(config, mode, **kwargs):
    """
    T14: IntervalReproduction (Ready-Set-Go).
 
    Two brief pulses (READY, SET) define an interval delta_t.
    After SET, the network must produce its own pulse exactly delta_t later.
 
    Inputs:  fixation (ch 0), pulses on SCALAR_IN_A (ch 5)
    Outputs: fixation (ch 0), produced pulse on SCALAR_OUT (ch 1)
 
    Based on Jazayeri & Shadlen (2010) Ready-Set-Go paradigm.
    """
    dt     = config['dt']
    rng    = config['rng']
    pulse_w = max(1, int(5 / dt))   # pulse width in timesteps (~5 ms)
 
    if mode == 'random':
        batch_size = kwargs['batch_size']
 
        # delta_t drawn per batch (same interval for all items in a batch).
        delta_t = int(rng.uniform(500, 1800) / dt)
 
        fix_dur   = int(rng.uniform(300, 700) / dt)
        ready_on  = fix_dur
        ready_off = ready_on + pulse_w
        interval  = delta_t
        set_on    = ready_off + interval
        set_off   = set_on + pulse_w
        prod_on   = set_off            # production window starts after SET
        # Target pulse centered at prod_on + delta_t.
        target_t  = prod_on + delta_t
        tdim      = target_t + int(rng.uniform(300, 700) / dt)
 
    elif mode == 'test':
        # Sample a few canonical intervals for evaluation.
        delta_ts   = [int(d / dt) for d in [600, 900, 1200, 1500]]
        batch_size = len(delta_ts)
 
        fix_dur    = int(500 / dt)
        # Build per-trial timing; for simplicity use the first delta_t for
        # shared epoch boundaries (test mode uses a single tdim).
        delta_t    = max(delta_ts)   # tdim must fit all trials
        ready_on   = fix_dur
        ready_off  = ready_on + pulse_w
        set_on     = ready_off + delta_t
        set_off    = set_on + pulse_w
        prod_on    = set_off
        target_t   = prod_on + delta_t
        tdim       = target_t + int(500 / dt)
 
        # delta_t is reused below — store per-trial values separately.
        delta_ts_arr = np.array(delta_ts)
 
    else:
        raise ValueError('Unknown mode: ' + str(mode))
 
    trial = Trial(config, tdim, batch_size)
    # Fixation stays on throughout (no go cue in the traditional sense).
    trial.add('fix_in')
    trial.add('fix_out')
 
    if mode == 'random':
        # READY pulse.
        trial.x[ready_on:ready_off, :, SCALAR_IN_A] = 1.0
        # SET pulse.
        trial.x[set_on:set_off, :, SCALAR_IN_A] = 1.0
        # Target: brief pulse of width pulse_w centered at target_t.
        t0 = max(0, target_t - pulse_w)
        t1 = min(tdim, target_t + pulse_w)
        trial.y[t0:t1, :, SCALAR_OUT] = 1.0
 
        check_ons = prod_on
        trial.add_c_mask(pre_offs=prod_on, post_ons=check_ons)
 
        trial.epochs = {
            'fix1'    : (None, ready_on),
            'ready'   : (ready_on, ready_off),
            'interval': (ready_off, set_on),
            'set'     : (set_on, set_off),
            'prod'    : (set_off, None),
        }
 
    else:  # test mode — per-trial timing
        for i, dti in enumerate(delta_ts_arr):
            r_on  = fix_dur
            r_off = r_on + pulse_w
            s_on  = r_off + dti
            s_off = s_on + pulse_w
            tgt   = s_off + dti
            trial.x[r_on:r_off, i, SCALAR_IN_A] = 1.0
            trial.x[s_on:s_off, i, SCALAR_IN_A] = 1.0
            t0 = max(0, tgt - pulse_w)
            t1 = min(tdim, tgt + pulse_w)
            trial.y[t0:t1, i, SCALAR_OUT] = 1.0
 
        check_ons = prod_on
        trial.add_c_mask(pre_offs=prod_on, post_ons=check_ons)
 
        trial.epochs = {
            'fix1'    : (None, fix_dur),
            'ready'   : (fix_dur, fix_dur + pulse_w),
            'interval': (fix_dur + pulse_w, set_on),
            'set'     : (set_on, set_off),
            'prod'    : (set_off, None),
        }
 
    trial.delta_t = delta_t
    return trial
 
 
# ---------------------------------------------------------------------------
# T15: PulseRateEstimation
# ---------------------------------------------------------------------------
 
def pulserateestimation(config, mode, **kwargs):
    """
    T15: PulseRateEstimation.
 
    A Poisson-ish pulse train at rate r arrives over a fixed observation
    window. After the window, output r / r_max as a scalar.
 
    Inputs:  fixation (ch 0), pulse train on SCALAR_IN_A (ch 5)
    Outputs: fixation (ch 0), scalar rate estimate on SCALAR_OUT (ch 1)
    """
    dt    = config['dt']
    rng   = config['rng']
    r_max = 0.30   # maximum rate in pulses/timestep
 
    if mode == 'random':
        batch_size = kwargs['batch_size']
        rates      = rng.uniform(0.05, r_max, size=batch_size)
 
        fix_dur    = int(rng.uniform(300, 700) / dt)
        obs_dur    = int(200 / dt)    # fixed 200-ts observation window
        fix_offs   = fix_dur + obs_dur
        tdim       = fix_offs + int(rng.uniform(300, 700) / dt)
 
    elif mode == 'test':
        rates      = np.linspace(0.05, r_max, 6)
        batch_size = len(rates)
        fix_dur    = int(500 / dt)
        obs_dur    = int(200 / dt)
        fix_offs   = fix_dur + obs_dur
        tdim       = fix_offs + int(500 / dt)
 
    else:
        raise ValueError('Unknown mode: ' + str(mode))
 
    obs_on  = fix_dur
    obs_off = fix_dur + obs_dur
 
    trial = Trial(config, tdim, batch_size)
    trial.add('fix_in', offs=fix_offs)
    trial.add('fix_out', offs=fix_offs)
 
    # Generate Poisson pulse train for each trial independently.
    for i in range(batch_size):
        for t in range(obs_on, obs_off):
            if rng.rand() < rates[i]:
                # Brief pulse of width 3 ts.
                trial.x[t:min(t + 3, obs_off), i, SCALAR_IN_A] = 1.0
 
    # Scalar rate output held for entire response epoch.
    for i in range(batch_size):
        trial.y[fix_offs:, i, SCALAR_OUT] = rates[i] / r_max
 
    check_ons = fix_offs + int(100 / dt)
    trial.add_c_mask(pre_offs=fix_offs, post_ons=check_ons)
 
    trial.epochs = {
        'fix1'   : (None, obs_on),
        'observe': (obs_on, obs_off),
        'go1'    : (fix_offs, None),
    }
 
    trial.rates = rates
    return trial
 
 
# ---------------------------------------------------------------------------
# T16: RhythmGeneration
# ---------------------------------------------------------------------------
 
def rhythmgeneration(config, mode, **kwargs):
    """
    T16: RhythmGeneration.
 
    A brief scalar cue on SCALAR_IN_A specifies a frequency f (normalized).
    The network must then sustain a sinusoidal output at that frequency.
    Requires limit cycle dynamics in the RNN.
 
    Inputs:  fixation (ch 0), frequency cue on SCALAR_IN_A (ch 5)
    Outputs: fixation (ch 0), sin(2*pi*f*t) on SCALAR_OUT (ch 1)
 
    Note: fixation drops during the SUSTAIN phase — we treat SUSTAIN as
    the response epoch. The network must self-sustain the oscillation.
    """
    dt    = config['dt']
    rng   = config['rng']
    f_min = 0.02   # cycles per timestep
    f_max = 0.08
 
    if mode == 'random':
        batch_size = kwargs['batch_size']
        freqs      = rng.uniform(f_min, f_max, size=batch_size)
 
        fix_dur    = int(rng.uniform(300, 700) / dt)
        cue_dur    = int(rng.uniform(200, 400) / dt)
        sustain_dur= int(rng.uniform(2000, 4000) / dt)
        cue_off    = fix_dur + cue_dur
        tdim       = cue_off + sustain_dur
 
    elif mode == 'test':
        freqs      = np.linspace(f_min, f_max, 4)
        batch_size = len(freqs)
        fix_dur    = int(500 / dt)
        cue_dur    = int(300 / dt)
        sustain_dur= int(3000 / dt)
        cue_off    = fix_dur + cue_dur
        tdim       = cue_off + sustain_dur
 
    else:
        raise ValueError('Unknown mode: ' + str(mode))
 
    trial = Trial(config, tdim, batch_size)
 
    # Fixation held during fix + cue; drops at sustain onset.
    trial.add('fix_in', offs=cue_off)
    trial.add('fix_out', offs=cue_off)
 
    # Frequency cue as normalized scalar during CUE epoch.
    for i in range(batch_size):
        trial.x[fix_dur:cue_off, i, SCALAR_IN_A] = freqs[i] / f_max
 
    # Target sinusoid during SUSTAIN epoch.
    for i in range(batch_size):
        for t in range(cue_off, tdim):
            phase = 2 * np.pi * freqs[i] * (t - cue_off)
            trial.y[t, i, SCALAR_OUT] = np.sin(phase)
 
    check_ons = cue_off + int(100 / dt)
    trial.add_c_mask(pre_offs=cue_off, post_ons=check_ons)
 
    trial.epochs = {
        'fix1'   : (None, fix_dur),
        'cue'    : (fix_dur, cue_off),
        'sustain': (cue_off, None),
    }
 
    trial.freqs = freqs
    return trial
 
 
# ---------------------------------------------------------------------------
# T17: SequenceRecall
# ---------------------------------------------------------------------------
 
def sequencerecall(config, mode, **kwargs):
    """
    T17: SequenceRecall.
 
    See K angles shown sequentially. After a delay, reproduce them in
    the same order, one after another.
 
    Inputs:  fixation (ch 0), angles on modality 1 (ring 1, ch 1-2)
    Outputs: fixation (ch 0), angles reproduced sequentially (ch 1-2)
    """
    dt  = config['dt']
    rng = config['rng']
 
    if mode == 'random':
        batch_size = kwargs['batch_size']
        K          = rng.choice([3, 4])
        angles     = rng.uniform(0, 2 * np.pi, (batch_size, K))
 
        fix_dur    = int(rng.uniform(300, 700) / dt)
        item_dur   = int(rng.uniform(400, 600) / dt)
        delay_dur  = int(rng.uniform(500, 1500) / dt)
        resp_dur   = int(rng.uniform(400, 600) / dt)  # per item
 
        enc_end    = fix_dur + K * item_dur
        fix_offs   = enc_end + delay_dur
        tdim       = fix_offs + K * resp_dur
 
    elif mode == 'test':
        K          = 3
        batch_size = 4
        angles     = np.tile(
            np.linspace(0, 2 * np.pi, K, endpoint=False), (batch_size, 1))
        angles    += rng.uniform(0, 0.5, (batch_size, K))
 
        fix_dur    = int(500 / dt)
        item_dur   = int(500 / dt)
        delay_dur  = int(1000 / dt)
        resp_dur   = int(500 / dt)
 
        enc_end    = fix_dur + K * item_dur
        fix_offs   = enc_end + delay_dur
        tdim       = fix_offs + K * resp_dur
 
    else:
        raise ValueError('Unknown mode: ' + str(mode))
 
    trial = Trial(config, tdim, batch_size)
    trial.add('fix_in', offs=fix_offs)
    trial.add('fix_out', offs=fix_offs)
 
    # Encoding: show each angle in turn on modality 1.
    for k in range(K):
        item_on  = fix_dur + k * item_dur
        item_off = item_on + item_dur
        trial.add('stim', angles[:, k], ons=item_on, offs=item_off, mods=1)
 
    # Response: reproduce each angle in the same order.
    for k in range(K):
        resp_on  = fix_offs + k * resp_dur
        resp_off = resp_on + resp_dur
        trial.add('out', angles[:, k], ons=resp_on, offs=resp_off)
 
    check_ons = fix_offs + int(100 / dt)
    trial.add_c_mask(pre_offs=fix_offs, post_ons=check_ons)
 
    trial.epochs = {
        'fix1'  : (None, fix_dur),
        'items' : (fix_dur, enc_end),
        'delay1': (enc_end, fix_offs),
        'resp'  : (fix_offs, None),
    }
 
    trial.K      = K
    trial.angles = angles
    return trial
 
 
# ---------------------------------------------------------------------------
# T18: Toggle
# ---------------------------------------------------------------------------
 
def toggle(config, mode, **kwargs):
    """
    T18: Toggle (bistable flip-flop).
 
    A stream of brief pulses arrives at irregular intervals on SCALAR_IN_A.
    Each pulse flips the output between -1 and +1.
    No explicit response epoch — the output is continuous throughout.
 
    Requires bistable dynamics in the RNN.
 
    Inputs:  fixation (ch 0), pulses on SCALAR_IN_A (ch 5)
    Outputs: fixation (ch 0), toggle state {-1, +1} on SCALAR_OUT (ch 1)
    """
    dt      = config['dt']
    rng     = config['rng']
 
    if mode == 'random':
        batch_size   = kwargs['batch_size']
        fix_dur      = int(rng.uniform(300, 700) / dt)
        stream_dur   = int(rng.uniform(2000, 4000) / dt)
        tdim         = fix_dur + stream_dur
        pulse_rate   = rng.uniform(0.01, 0.04)   # pulses per timestep
 
    elif mode == 'test':
        batch_size   = 4
        fix_dur      = int(500 / dt)
        stream_dur   = int(3000 / dt)
        tdim         = fix_dur + stream_dur
        pulse_rate   = 0.02
 
    else:
        raise ValueError('Unknown mode: ' + str(mode))
 
    trial = Trial(config, tdim, batch_size)
    trial.add('fix_in', offs=fix_dur)
    trial.add('fix_out', offs=fix_dur)
 
    pulse_w = max(1, int(4 / dt))
 
    for i in range(batch_size):
        # Initial state drawn randomly.
        state = rng.choice([-1.0, 1.0])
        trial.y[fix_dur:, i, SCALAR_OUT] = state
 
        t = fix_dur
        while t < tdim - pulse_w:
            # Poisson-ish inter-pulse interval.
            if rng.rand() < pulse_rate:
                # Place pulse.
                p_end = min(t + pulse_w, tdim)
                trial.x[t:p_end, i, SCALAR_IN_A] = 1.0
                # Flip state after pulse.
                state = -state
                # Update output from pulse end onward.
                trial.y[p_end:, i, SCALAR_OUT] = state
                t += pulse_w + max(1, int(rng.uniform(20, 70) / dt))
            else:
                t += 1
 
    # Cost mask covers the whole stream; pre_offs at fix end.
    check_ons = fix_dur + int(100 / dt)
    trial.add_c_mask(pre_offs=fix_dur, post_ons=check_ons)
 
    trial.epochs = {
        'fix1'  : (None, fix_dur),
        'stream': (fix_dur, None),
    }
 
    return trial
 
 
# ---------------------------------------------------------------------------
# T19: ConditionalToggle
# ---------------------------------------------------------------------------
 
def conditionaltoggle(config, mode, **kwargs):
    """
    T19: ConditionalToggle.
 
    Two independent pulse streams arrive on SCALAR_IN_A and SCALAR_IN_B.
    Pulses on SCALAR_IN_A toggle output channel SCALAR_OUT independently
    from pulses on SCALAR_IN_B, which toggle a second output channel
    (y[:, :, 2], reusing the cos slot since no circular output needed here).
 
    Requires two independent bistable units gated by input channel.
 
    Inputs:  fixation (ch 0), pulses on SCALAR_IN_A (ch 5), SCALAR_IN_B (ch 6)
    Outputs: fixation (ch 0), state A on ch 1, state B on ch 2
    """
    dt    = config['dt']
    rng   = config['rng']
 
    SCALAR_OUT_B = 2  # second scalar output reuses the cos slot
 
    if mode == 'random':
        batch_size  = kwargs['batch_size']
        fix_dur     = int(rng.uniform(300, 700) / dt)
        stream_dur  = int(rng.uniform(2000, 4000) / dt)
        tdim        = fix_dur + stream_dur
        rate_a      = rng.uniform(0.01, 0.03)
        rate_b      = rng.uniform(0.01, 0.03)
 
    elif mode == 'test':
        batch_size  = 4
        fix_dur     = int(500 / dt)
        stream_dur  = int(3000 / dt)
        tdim        = fix_dur + stream_dur
        rate_a      = 0.02
        rate_b      = 0.015
 
    else:
        raise ValueError('Unknown mode: ' + str(mode))
 
    trial = Trial(config, tdim, batch_size)
    trial.add('fix_in', offs=fix_dur)
    trial.add('fix_out', offs=fix_dur)
 
    pulse_w = max(1, int(4 / dt))
 
    for i in range(batch_size):
        state_a = rng.choice([-1.0, 1.0])
        state_b = rng.choice([-1.0, 1.0])
        trial.y[fix_dur:, i, SCALAR_OUT]   = state_a
        trial.y[fix_dur:, i, SCALAR_OUT_B] = state_b
 
        # Channel A pulses.
        t = fix_dur
        while t < tdim - pulse_w:
            if rng.rand() < rate_a:
                p_end = min(t + pulse_w, tdim)
                trial.x[t:p_end, i, SCALAR_IN_A] = 1.0
                state_a = -state_a
                trial.y[p_end:, i, SCALAR_OUT] = state_a
                t += pulse_w + max(1, int(rng.uniform(20, 70) / dt))
            else:
                t += 1
 
        # Channel B pulses (independent).
        t = fix_dur
        while t < tdim - pulse_w:
            if rng.rand() < rate_b:
                p_end = min(t + pulse_w, tdim)
                trial.x[t:p_end, i, SCALAR_IN_B] = 1.0
                state_b = -state_b
                trial.y[p_end:, i, SCALAR_OUT_B] = state_b
                t += pulse_w + max(1, int(rng.uniform(20, 70) / dt))
            else:
                t += 1
 
    check_ons = fix_dur + int(100 / dt)
    trial.add_c_mask(pre_offs=fix_dur, post_ons=check_ons)
 
    trial.epochs = {
        'fix1'  : (None, fix_dur),
        'stream': (fix_dur, None),
    }
 
    return trial
 
 
# ---------------------------------------------------------------------------
# T20: CueResponseAssoc
# ---------------------------------------------------------------------------
 
def cueassoc(config, mode, **kwargs):
    """
    T20 prototype: CueResponseAssoc.
 
    Trial structure: FIX -> CUE -> DELAY -> GO
 
    Simplified supervised version: mapping is fixed across all lifetimes.
    Each cue identity is represented as a discrete angle on modality 1.
    Full T20 requires a fresh random mapping per lifetime (meta-learning phase).
 
    Inputs:  fixation (ch 0), cue angle on modality 1 (ch 1-2)
    Outputs: fixation (ch 0), associated response angle (ch 1-2)
    """
    dt     = config['dt']
    rng    = config['rng']
    n_cues = kwargs.get('n_cues', 8)
 
    cue_locs_all      = np.linspace(0, 2 * np.pi, n_cues, endpoint=False)
    # Fixed mapping: cue -> cue + pi/2. Replace with random per lifetime for full T20.
    response_locs_all = (cue_locs_all + np.pi / 2) % (2 * np.pi)
 
    if mode == 'random':
        batch_size    = kwargs['batch_size']
        cue_ids       = rng.choice(n_cues, size=batch_size)
        cue_locs      = cue_locs_all[cue_ids]
        response_locs = response_locs_all[cue_ids]
 
        cue_ons  = int(rng.uniform(300, 700) / dt)
        cue_offs = cue_ons + int(rng.uniform(300, 800) / dt)
        fix_offs = cue_offs + int(rng.uniform(200, 800) / dt)
        tdim     = fix_offs + int(rng.uniform(300, 700) / dt)
 
    elif mode == 'test':
        batch_size    = n_cues
        cue_ids       = np.arange(n_cues)
        cue_locs      = cue_locs_all
        response_locs = response_locs_all
 
        cue_ons  = int(500 / dt)
        cue_offs = int(1000 / dt)
        fix_offs = int(1500 / dt)
        tdim     = int(2000 / dt)
 
    elif mode == 'psychometric':
        p             = kwargs['params']
        cue_ids       = np.array(p['cue_ids'])
        batch_size    = len(cue_ids)
        cue_locs      = cue_locs_all[cue_ids]
        response_locs = response_locs_all[cue_ids]
 
        cue_ons  = int(p.get('cue_ons', 500) / dt)
        cue_offs = int(p.get('cue_offs', 1000) / dt)
        fix_offs = int(p.get('fix_offs', 1500) / dt)
        tdim     = int(p.get('tdim', 2000) / dt)
 
    else:
        raise ValueError('Unknown mode: ' + str(mode))
 
    check_ons = fix_offs + int(100 / dt)
 
    trial = Trial(config, tdim, batch_size)
    trial.add('fix_in', offs=fix_offs)
    trial.add('stim', cue_locs, ons=cue_ons, offs=cue_offs, mods=1)
    trial.add('fix_out', offs=fix_offs)
    trial.add('out', response_locs, ons=fix_offs)
    trial.add_c_mask(pre_offs=fix_offs, post_ons=check_ons)
 
    trial.epochs = {
        'fix1'  : (None, cue_ons),
        'cue1'  : (cue_ons, cue_offs),
        'delay1': (cue_offs, fix_offs),
        'go1'   : (fix_offs, None),
    }
 
    trial.cue_ids      = cue_ids
    trial.cue_locs     = cue_locs
    trial.response_locs = response_locs
    return trial
 
 
# ---------------------------------------------------------------------------
# T21: PairedAssociation 
# ---------------------------------------------------------------------------
 
def pairedassoc(config, mode, **kwargs):
    """
    T21 prototype: PairedAssociation.
 
    Within one trial:
        PAIR_A: cue A (modality 1) + angle A (modality 2) shown simultaneously
        PAIR_B: cue B + angle B
        PROBE:  one cue shown alone
        RESP:   output angle paired with the probed cue
 
    Tests within-trial one-shot associative binding.
    """
    dt     = config['dt']
    rng    = config['rng']
    n_cues = kwargs.get('n_cues', 8)
 
    cue_locs_all = np.linspace(0, 2 * np.pi, n_cues, endpoint=False)
 
    if mode == 'random':
        batch_size  = kwargs['batch_size']
        cue_ids_a   = rng.choice(n_cues, size=batch_size)
        cue_ids_b   = rng.choice(n_cues, size=batch_size)
        same        = cue_ids_b == cue_ids_a
        cue_ids_b[same] = (cue_ids_b[same] + 1) % n_cues
 
        cue_locs_a  = cue_locs_all[cue_ids_a]
        cue_locs_b  = cue_locs_all[cue_ids_b]
        angle_a     = rng.uniform(0, 2 * np.pi, batch_size)
        angle_b     = rng.uniform(0, 2 * np.pi, batch_size)
        probe_is_a  = rng.choice([0, 1], size=batch_size).astype(bool)
        probe_locs  = np.where(probe_is_a, cue_locs_a, cue_locs_b)
        response_locs = np.where(probe_is_a, angle_a, angle_b)
 
        fix_dur     = int(rng.uniform(300, 700) / dt)
        pair_dur    = int(rng.uniform(400, 800) / dt)
        delay1_dur  = int(rng.uniform(300, 800) / dt)
        delay2_dur  = int(rng.uniform(300, 800) / dt)
        probe_dur   = int(rng.uniform(300, 600) / dt)
        resp_dur    = int(rng.uniform(300, 700) / dt)
 
    elif mode == 'test':
        batch_size  = n_cues
        cue_ids_a   = np.arange(n_cues)
        cue_ids_b   = (cue_ids_a + 1) % n_cues
        cue_locs_a  = cue_locs_all[cue_ids_a]
        cue_locs_b  = cue_locs_all[cue_ids_b]
        angle_a     = (cue_locs_a + np.pi / 4) % (2 * np.pi)
        angle_b     = (cue_locs_b + np.pi / 2) % (2 * np.pi)
        probe_is_a  = np.arange(n_cues) % 2 == 0
        probe_locs  = np.where(probe_is_a, cue_locs_a, cue_locs_b)
        response_locs = np.where(probe_is_a, angle_a, angle_b)
 
        fix_dur     = int(500 / dt)
        pair_dur    = int(500 / dt)
        delay1_dur  = int(500 / dt)
        delay2_dur  = int(500 / dt)
        probe_dur   = int(500 / dt)
        resp_dur    = int(500 / dt)
 
    else:
        raise ValueError('Unknown mode: ' + str(mode))
 
    pair_a_on  = fix_dur
    pair_a_off = pair_a_on  + pair_dur
    pair_b_on  = pair_a_off + delay1_dur
    pair_b_off = pair_b_on  + pair_dur
    probe_on   = pair_b_off + delay2_dur
    probe_off  = probe_on   + probe_dur
    fix_offs   = probe_off
    tdim       = fix_offs   + resp_dur
 
    check_ons  = fix_offs + int(100 / dt)
 
    trial = Trial(config, tdim, batch_size)
    trial.add('fix_in', offs=fix_offs)
    trial.add('stim', cue_locs_a, ons=pair_a_on,  offs=pair_a_off, mods=1)
    trial.add('stim', angle_a,    ons=pair_a_on,  offs=pair_a_off, mods=2)
    trial.add('stim', cue_locs_b, ons=pair_b_on,  offs=pair_b_off, mods=1)
    trial.add('stim', angle_b,    ons=pair_b_on,  offs=pair_b_off, mods=2)
    trial.add('stim', probe_locs, ons=probe_on,   offs=probe_off,  mods=1)
    trial.add('fix_out', offs=fix_offs)
    trial.add('out', response_locs, ons=fix_offs)
    trial.add_c_mask(pre_offs=fix_offs, post_ons=check_ons)
 
    trial.epochs = {
        'fix1'  : (None, pair_a_on),
        'pair_a': (pair_a_on,  pair_a_off),
        'delay1': (pair_a_off, pair_b_on),
        'pair_b': (pair_b_on,  pair_b_off),
        'delay2': (pair_b_off, probe_on),
        'probe' : (probe_on,   probe_off),
        'go1'   : (fix_offs,   None),
    }
 
    trial.cue_ids_a    = cue_ids_a
    trial.cue_ids_b    = cue_ids_b
    trial.probe_is_a   = probe_is_a
    trial.response_locs = response_locs
    return trial
 
 # T22: ReversalLearning
# ---------------------------------------------------------------------------
 
def reversallearning(config, mode, **kwargs):
    """
    T22: ReversalLearning.
 
    Like CueResponseAssoc (T20) but the cue->angle mapping reverses at an
    unknown trial t_rev within a lifetime. The network must detect the change
    and relearn the new mapping.
 
    In the supervised proxy here, 'reversal' is signaled implicitly by a
    flag on SCALAR_IN_B (1 = post-reversal, 0 = pre-reversal). This lets
    the network learn to switch mappings within a single trial. The full
    across-lifetime version (where reversal is not flagged) requires the
    meta-learning setup.
 
    Inputs:  fixation (ch 0), cue angle on modality 1 (ch 1-2),
             reversal flag on SCALAR_IN_B (ch 6)
    Outputs: fixation (ch 0), associated response angle (ch 1-2)
    """
    dt     = config['dt']
    rng    = config['rng']
    n_cues = kwargs.get('n_cues', 4)
 
    cue_locs_all          = np.linspace(0, 2 * np.pi, n_cues, endpoint=False)
    response_locs_pre     = (cue_locs_all + np.pi / 2) % (2 * np.pi)
    # Post-reversal: mapping flips by pi.
    response_locs_post    = (response_locs_pre + np.pi) % (2 * np.pi)
 
    if mode == 'random':
        batch_size    = kwargs['batch_size']
        cue_ids       = rng.choice(n_cues, size=batch_size)
        cue_locs      = cue_locs_all[cue_ids]
        post_reversal = rng.choice([0, 1], size=batch_size).astype(bool)
        response_locs = np.where(
            post_reversal,
            response_locs_post[cue_ids],
            response_locs_pre[cue_ids])
 
        cue_ons  = int(rng.uniform(300, 700) / dt)
        cue_offs = cue_ons + int(rng.uniform(300, 800) / dt)
        fix_offs = cue_offs + int(rng.uniform(200, 800) / dt)
        tdim     = fix_offs + int(rng.uniform(300, 700) / dt)
 
    elif mode == 'test':
        # One trial per cue x {pre, post} reversal.
        batch_size    = n_cues * 2
        cue_ids_base  = np.tile(np.arange(n_cues), 2)
        post_reversal = np.array([False] * n_cues + [True] * n_cues)
        cue_locs      = cue_locs_all[cue_ids_base]
        response_locs = np.where(
            post_reversal,
            response_locs_post[cue_ids_base],
            response_locs_pre[cue_ids_base])
 
        cue_ons  = int(500 / dt)
        cue_offs = int(1000 / dt)
        fix_offs = int(1500 / dt)
        tdim     = int(2000 / dt)
 
    else:
        raise ValueError('Unknown mode: ' + str(mode))
 
    check_ons = fix_offs + int(100 / dt)
 
    trial = Trial(config, tdim, batch_size)
    trial.add('fix_in', offs=fix_offs)
    trial.add('stim', cue_locs, ons=cue_ons, offs=cue_offs, mods=1)
 
    # Reversal flag on SCALAR_IN_B throughout cue period.
    for i in range(batch_size):
        if post_reversal[i]:
            trial.x[cue_ons:cue_offs, i, SCALAR_IN_B] = 1.0
 
    trial.add('fix_out', offs=fix_offs)
    trial.add('out', response_locs, ons=fix_offs)
    trial.add_c_mask(pre_offs=fix_offs, post_ons=check_ons)
 
    trial.epochs = {
        'fix1'  : (None, cue_ons),
        'cue1'  : (cue_ons, cue_offs),
        'delay1': (cue_offs, fix_offs),
        'go1'   : (fix_offs, None),
    }
 
    trial.cue_ids      = cue_ids_base if mode == 'test' else cue_ids
    trial.post_reversal = post_reversal
    trial.response_locs = response_locs
    return trial
 
 
# ---------------------------------------------------------------------------
# T23: OnlineLinearReg
# ---------------------------------------------------------------------------
 
def onlinelinearreg(config, mode, **kwargs):
    """
    T23: OnlineLinearReg.
 
    Within one trial: K examples of (x, y) where y = a*x + b, shown
    sequentially. Then a query x_q; predict y_q = a*x_q + b.
 
    The function parameters (a, b) are fresh every trial.
    The network must infer a and b from the examples and generalize.
 
    Inputs:  fixation (ch 0), x on SCALAR_IN_A (ch 5), y on SCALAR_IN_B (ch 6)
             During probe: x_q on SCALAR_IN_A, SCALAR_IN_B = 0.
    Outputs: fixation (ch 0), predicted y_q on SCALAR_OUT_A (ch 1)
    """
    dt  = config['dt']
    rng = config['rng']
 
    if mode == 'random':
        batch_size = kwargs['batch_size']
        K          = rng.randint(5, 8)   # number of examples
        a          = rng.uniform(-1.0, 1.0, batch_size)
        b          = rng.uniform(-0.5, 0.5, batch_size)
 
        fix_dur    = int(rng.uniform(300, 600) / dt)
        ex_dur     = int(rng.uniform(300, 450) / dt)   # per example
        gap_dur    = int(80 / dt)
        probe_dur  = int(rng.uniform(350, 550) / dt)
        resp_dur   = int(rng.uniform(300, 700) / dt)
 
    elif mode == 'test':
        batch_size = 8
        K          = 6
        a          = np.linspace(-0.8, 0.8, batch_size)
        b          = np.zeros(batch_size)
 
        fix_dur    = int(500 / dt)
        ex_dur     = int(400 / dt)
        gap_dur    = int(80 / dt)
        probe_dur  = int(400 / dt)
        resp_dur   = int(500 / dt)
 
    else:
        raise ValueError('Unknown mode: ' + str(mode))
 
    examples_dur = K * (ex_dur + gap_dur)
    fix_offs     = fix_dur + examples_dur + probe_dur
    tdim         = fix_offs + resp_dur
 
    trial = Trial(config, tdim, batch_size)
    trial.add('fix_in', offs=fix_offs)
    trial.add('fix_out', offs=fix_offs)
 
    # Generate and place K examples.
    x_vals = rng.uniform(-0.8, 0.8, (batch_size, K))
    for k in range(K):
        t0 = fix_dur + k * (ex_dur + gap_dur)
        t1 = t0 + ex_dur
        for i in range(batch_size):
            y_val = a[i] * x_vals[i, k] + b[i]
            trial.x[t0:t1, i, SCALAR_IN_A] = x_vals[i, k]
            trial.x[t0:t1, i, SCALAR_IN_B] = np.clip(y_val, -1.0, 1.0)
 
    # Query: x_q only, no y.
    x_q   = rng.uniform(-0.8, 0.8, batch_size)
    y_q   = np.clip(a * x_q + b, -1.0, 1.0)
    probe_on  = fix_dur + examples_dur
    probe_off = probe_on + probe_dur
    for i in range(batch_size):
        trial.x[probe_on:probe_off, i, SCALAR_IN_A] = x_q[i]
 
    # Target: predicted y_q held over response epoch.
    for i in range(batch_size):
        trial.y[fix_offs:, i, SCALAR_OUT_A] = y_q[i]
 
    check_ons = fix_offs + int(100 / dt)
    trial.add_c_mask(pre_offs=fix_offs, post_ons=check_ons)
 
    trial.epochs = {
        'fix1'    : (None, fix_dur),
        'examples': (fix_dur, probe_on),
        'probe'   : (probe_on, probe_off),
        'go1'     : (fix_offs, None),
    }
 
    trial.a   = a
    trial.b   = b
    trial.x_q = x_q
    trial.y_q = y_q
    return trial
 
 
# ---------------------------------------------------------------------------
# T24: OnlineNonlinearReg
# ---------------------------------------------------------------------------
 
def onlinenonlinearreg(config, mode, **kwargs):
    """
    T24: OnlineNonlinearReg.
 
    Like T23 but y = sin(omega * x + phi).
    The network must infer omega and phi from K examples and generalize.
 
    Inputs/Outputs: same channel layout as T23.
    """
    dt  = config['dt']
    rng = config['rng']
 
    if mode == 'random':
        batch_size = kwargs['batch_size']
        K          = rng.randint(7, 11)
        omega      = rng.uniform(0.8, 2.5, batch_size)
        phi        = rng.uniform(0, 2 * np.pi, batch_size)
 
        fix_dur    = int(rng.uniform(300, 600) / dt)
        ex_dur     = int(rng.uniform(250, 350) / dt)
        gap_dur    = int(60 / dt)
        probe_dur  = int(rng.uniform(350, 550) / dt)
        resp_dur   = int(rng.uniform(300, 700) / dt)
 
    elif mode == 'test':
        batch_size = 6
        K          = 8
        omega      = np.linspace(0.8, 2.4, batch_size)
        phi        = np.zeros(batch_size)
 
        fix_dur    = int(500 / dt)
        ex_dur     = int(300 / dt)
        gap_dur    = int(60 / dt)
        probe_dur  = int(400 / dt)
        resp_dur   = int(500 / dt)
 
    else:
        raise ValueError('Unknown mode: ' + str(mode))
 
    examples_dur = K * (ex_dur + gap_dur)
    fix_offs     = fix_dur + examples_dur + probe_dur
    tdim         = fix_offs + resp_dur
 
    trial = Trial(config, tdim, batch_size)
    trial.add('fix_in', offs=fix_offs)
    trial.add('fix_out', offs=fix_offs)
 
    x_vals = rng.uniform(-1.0, 1.0, (batch_size, K))
    for k in range(K):
        t0 = fix_dur + k * (ex_dur + gap_dur)
        t1 = t0 + ex_dur
        for i in range(batch_size):
            y_val = np.sin(omega[i] * x_vals[i, k] + phi[i])
            trial.x[t0:t1, i, SCALAR_IN_A] = x_vals[i, k]
            trial.x[t0:t1, i, SCALAR_IN_B] = y_val   # already in [-1, 1]
 
    x_q       = rng.uniform(-1.0, 1.0, batch_size)
    y_q       = np.sin(omega * x_q + phi)
    probe_on  = fix_dur + examples_dur
    probe_off = probe_on + probe_dur
    for i in range(batch_size):
        trial.x[probe_on:probe_off, i, SCALAR_IN_A] = x_q[i]
 
    for i in range(batch_size):
        trial.y[fix_offs:, i, SCALAR_OUT_A] = y_q[i]
 
    check_ons = fix_offs + int(100 / dt)
    trial.add_c_mask(pre_offs=fix_offs, post_ons=check_ons)
 
    trial.epochs = {
        'fix1'    : (None, fix_dur),
        'examples': (fix_dur, probe_on),
        'probe'   : (probe_on, probe_off),
        'go1'     : (fix_offs, None),
    }
 
    trial.omega = omega
    trial.phi   = phi
    trial.x_q   = x_q
    trial.y_q   = y_q
    return trial
 
 
# ---------------------------------------------------------------------------
# T25: FewShotClassif
# ---------------------------------------------------------------------------
 
def fewshotclassif(config, mode, **kwargs):
    """
    T25: FewShotClassif.
 
    K classes x M examples each (support set), then a query.
    Each example: a 4-dim feature vector + scalar class label.
    Query: feature vector only. Output: class label as scalar.
 
    Feature vectors are packed into ring 1 + ring 2 channels (ch 1-4)
    as raw floats (not sin/cos encoded). Class label on SCALAR_IN_A (ch 5).
    Query feature on ch 1-4, SCALAR_IN_A = 0 during query.
 
    Output: class index (1..K) / K as scalar on SCALAR_OUT_A (ch 1).
 
    NOTE: This bypasses the ring encoder in trial.add('stim'). We write
    directly to trial.x for the feature dimensions. When expanding n_input
    in network.py for dedicated cue channels 7-10, update accordingly.
    """
    dt  = config['dt']
    rng = config['rng']
    K   = kwargs.get('K', 3)   # number of classes
    M   = kwargs.get('M', 3)   # examples per class
 
    if mode == 'random':
        batch_size = kwargs['batch_size']
 
        fix_dur    = int(rng.uniform(300, 600) / dt)
        ex_dur     = int(rng.uniform(300, 400) / dt)
        gap_dur    = int(60 / dt)
        delay_dur  = int(rng.uniform(400, 700) / dt)
        query_dur  = int(rng.uniform(400, 550) / dt)
        resp_dur   = int(rng.uniform(300, 700) / dt)
 
    elif mode == 'test':
        batch_size = K * M
        fix_dur    = int(500 / dt)
        ex_dur     = int(400 / dt)
        gap_dur    = int(60 / dt)
        delay_dur  = int(500 / dt)
        query_dur  = int(400 / dt)
        resp_dur   = int(500 / dt)
 
    else:
        raise ValueError('Unknown mode: ' + str(mode))
 
    support_dur  = K * M * (ex_dur + gap_dur)
    query_on     = fix_dur + support_dur + delay_dur
    fix_offs     = query_on + query_dur
    tdim         = fix_offs + resp_dur
 
    trial = Trial(config, tdim, batch_size)
    trial.add('fix_in', offs=fix_offs)
    trial.add('fix_out', offs=fix_offs)
 
    # Generate class prototypes: 4-dim vectors in [-1, 1].
    for i in range(batch_size):
        prototypes = rng.uniform(-1, 1, (K, 4))
 
        # Build shuffled support set.
        support = []
        for k in range(K):
            for m in range(M):
                # Add small noise to prototype.
                feat = prototypes[k] + rng.uniform(-0.15, 0.15, 4)
                feat = np.clip(feat, -1, 1)
                support.append((feat, k))
        rng.shuffle(support)
 
        for idx, (feat, cls) in enumerate(support):
            t0 = fix_dur + idx * (ex_dur + gap_dur)
            t1 = t0 + ex_dur
            # Feature into channels 1-4 directly.
            trial.x[t0:t1, i, 1:5] = feat
            # Class label (1-indexed, normalized) on SCALAR_IN_A.
            trial.x[t0:t1, i, SCALAR_IN_A] = (cls + 1) / K
 
        # Query: random example from a randomly chosen class.
        query_cls  = rng.randint(0, K)
        query_feat = prototypes[query_cls] + rng.uniform(-0.15, 0.15, 4)
        query_feat = np.clip(query_feat, -1, 1)
        trial.x[query_on:fix_offs, i, 1:5] = query_feat
        # SCALAR_IN_A = 0 during query (no label given).
 
        # Target: class label as scalar.
        trial.y[fix_offs:, i, SCALAR_OUT_A] = (query_cls + 1) / K
 
    check_ons = fix_offs + int(100 / dt)
    trial.add_c_mask(pre_offs=fix_offs, post_ons=check_ons)
 
    trial.epochs = {
        'fix1'   : (None, fix_dur),
        'support': (fix_dur, fix_dur + support_dur),
        'delay1' : (fix_dur + support_dur, query_on),
        'query'  : (query_on, fix_offs),
        'go1'    : (fix_offs, None),
    }
 
    trial.K = K
    trial.M = M
    return trial
 
 
# ---------------------------------------------------------------------------
# T26: MemoryDM  (Compositional: memory + decision-making)
# ---------------------------------------------------------------------------
 
def memorydm(config, mode, **kwargs):
    """
    T26: MemoryDM.
 
    Store angle theta from a brief stimulus. Then observe two scalar
    evidence streams (A_A on SCALAR_IN_A, A_B on SCALAR_IN_B) simultaneously.
    Respond at theta if A_A > A_B, else respond at theta + pi.
 
    Composes MemoryPro (T03) and DM (T05).
    Good OOD test if held out during training.
 
    Inputs:  fixation (ch 0), theta on modality 1 (ch 1-2),
             evidence A on SCALAR_IN_A (ch 5), evidence B on SCALAR_IN_B (ch 6)
    Outputs: fixation (ch 0), response angle (ch 1-2)
    """
    dt  = config['dt']
    rng = config['rng']
 
    if mode == 'random':
        batch_size  = kwargs['batch_size']
        theta       = rng.uniform(0, 2 * np.pi, batch_size)
        A_A         = rng.uniform(0.3, 1.0, batch_size)
        A_B         = rng.uniform(0.3, 1.0, batch_size)
 
        fix_dur     = int(rng.uniform(300, 700) / dt)
        stim_dur    = int(rng.uniform(300, 800) / dt)
        delay_dur   = int(rng.uniform(800, 2000) / dt)
        ev_dur      = int(rng.uniform(800, 2000) / dt)
        resp_dur    = int(rng.uniform(300, 700) / dt)
 
    elif mode == 'test':
        n           = 8
        batch_size  = n
        theta       = np.linspace(0, 2 * np.pi, n, endpoint=False)
        A_A         = np.where(np.arange(n) % 2 == 0, 0.8, 0.4)
        A_B         = 1.2 - A_A
 
        fix_dur     = int(500 / dt)
        stim_dur    = int(500 / dt)
        delay_dur   = int(1200 / dt)
        ev_dur      = int(1200 / dt)
        resp_dur    = int(500 / dt)
 
    else:
        raise ValueError('Unknown mode: ' + str(mode))
 
    stim_on  = fix_dur
    stim_off = stim_on + stim_dur
    ev_on    = stim_off + delay_dur
    ev_off   = ev_on + ev_dur
    fix_offs = ev_off
    tdim     = fix_offs + resp_dur
 
    # Response: theta if A_A > A_B, else theta + pi.
    response_locs = np.where(A_A > A_B, theta, (theta + np.pi) % (2 * np.pi))
 
    trial = Trial(config, tdim, batch_size)
    trial.add('fix_in', offs=fix_offs)
    trial.add('stim', theta, ons=stim_on, offs=stim_off, mods=1)
 
    # Evidence scalars during evidence epoch.
    for i in range(batch_size):
        trial.x[ev_on:ev_off, i, SCALAR_IN_A] = A_A[i]
        trial.x[ev_on:ev_off, i, SCALAR_IN_B] = A_B[i]
 
    trial.add('fix_out', offs=fix_offs)
    trial.add('out', response_locs, ons=fix_offs)
 
    check_ons = fix_offs + int(100 / dt)
    trial.add_c_mask(pre_offs=fix_offs, post_ons=check_ons)
 
    trial.epochs = {
        'fix1'    : (None, stim_on),
        'stim1'   : (stim_on, stim_off),
        'delay1'  : (stim_off, ev_on),
        'evidence': (ev_on, ev_off),
        'go1'     : (fix_offs, None),
    }
 
    trial.theta         = theta
    trial.A_A           = A_A
    trial.A_B           = A_B
    trial.response_locs = response_locs
    return trial
 
 
# ---------------------------------------------------------------------------
# T27: CountAndRecall  (Compositional: counting + binding)
# ---------------------------------------------------------------------------
 
def countandrecall(config, mode, **kwargs):
    """
    T27: CountAndRecall.
 
    N pulses arrive on SCALAR_IN_A, each simultaneously paired with a brief
    angle on modality 1. After all pulses, a scalar probe on SCALAR_IN_B
    indicates which pulse number to recall (1..N normalized as i/N).
    Output: the angle of the i-th pulse.
 
    Strongest OOD test: composes counting (T13) and associative binding (T21).
 
    Inputs:  fixation (ch 0), pulse on SCALAR_IN_A (ch 5),
             angle on modality 1 (ch 1-2), probe on SCALAR_IN_B (ch 6)
    Outputs: fixation (ch 0), recalled angle (ch 1-2)
    """
    dt  = config['dt']
    rng = config['rng']
    N_max = 4
 
    if mode == 'random':
        batch_size = kwargs['batch_size']
        N          = rng.randint(2, N_max + 1, batch_size)
 
        fix_dur    = int(rng.uniform(300, 700) / dt)
        stream_dur = int(rng.uniform(2000, 3000) / dt)
        probe_dur  = int(rng.uniform(400, 600) / dt)
        resp_dur   = int(rng.uniform(300, 700) / dt)
 
    elif mode == 'test':
        Ns         = [2, 3, 4]
        batch_size = len(Ns)
        N          = np.array(Ns)
 
        fix_dur    = int(500 / dt)
        stream_dur = int(2500 / dt)
        probe_dur  = int(500 / dt)
        resp_dur   = int(500 / dt)
 
    else:
        raise ValueError('Unknown mode: ' + str(mode))
 
    stream_on  = fix_dur
    stream_off = stream_on + stream_dur
    probe_on   = stream_off
    probe_off  = probe_on + probe_dur
    fix_offs   = probe_off
    tdim       = fix_offs + resp_dur
 
    trial = Trial(config, tdim, batch_size)
    trial.add('fix_in', offs=fix_offs)
    trial.add('fix_out', offs=fix_offs)
 
    pulse_w   = max(1, int(5 / dt))
    angle_dur = max(1, int(15 / dt))
 
    for i in range(batch_size):
        n_i     = N[i] if hasattr(N, '__len__') else N
        angles  = rng.uniform(0, 2 * np.pi, n_i)
        recall  = rng.randint(0, n_i)
 
        # Place N pulses at spaced positions within stream window.
        spacing   = stream_dur // (n_i + 1)
        positions = [stream_on + spacing * (j + 1) for j in range(n_i)]
        positions = [min(p, stream_off - angle_dur - 5) for p in positions]
 
        for j, p in enumerate(positions):
            # Pulse on SCALAR_IN_A.
            trial.x[p:p + pulse_w, i, SCALAR_IN_A] = 1.0
            # Paired angle on modality 1 (written directly to ch 1-2).
            trial.x[p:p + angle_dur, i, 1] = np.sin(angles[j])
            trial.x[p:p + angle_dur, i, 2] = np.cos(angles[j])
 
        # Probe: which pulse to recall, as (recall+1)/n_i on SCALAR_IN_B.
        trial.x[probe_on:probe_off, i, SCALAR_IN_B] = (recall + 1) / n_i
 
        # Target: recalled angle on ch 1-2.
        trial.y[fix_offs:, i, 1] = np.sin(angles[recall])
        trial.y[fix_offs:, i, 2] = np.cos(angles[recall])
 
    check_ons = fix_offs + int(100 / dt)
    trial.add_c_mask(pre_offs=fix_offs, post_ons=check_ons)
 
    trial.epochs = {
        'fix1'  : (None, stream_on),
        'stream': (stream_on, stream_off),
        'probe' : (probe_on, probe_off),
        'go1'   : (fix_offs, None),
    }
 
    trial.N = N
    return trial
 
 
# ---------------------------------------------------------------------------
# T28: ConditionalRhythm  (Compositional: limit cycle + context gating)
# ---------------------------------------------------------------------------
 
def conditionalrhythm(config, mode, **kwargs):
    """
    T28: ConditionalRhythm.
 
    A frequency cue f1 on SCALAR_IN_A sets an initial rhythm.
    A switch pulse on SCALAR_IN_B mid-trial commands a change to frequency f2
    (which is not explicitly cued — the network must remember it was set at
    the start via a second scalar, or infer from context).
 
    Simplified here: f2 is given at trial start on SCALAR_IN_B alongside f1
    on SCALAR_IN_A, then SCALAR_IN_B serves as the switch pulse.
 
    Inputs:  fixation (ch 0), f1 on SCALAR_IN_A (ch 5), f2 on SCALAR_IN_B (ch 6)
             during CUE_1; switch pulse on SCALAR_IN_B during PHASE_1.
    Outputs: fixation (ch 0), sin(2*pi*f*t) on SCALAR_OUT_A (ch 1)
    """
    dt    = config['dt']
    rng   = config['rng']
    f_min = 0.02
    f_max = 0.10
 
    if mode == 'random':
        batch_size  = kwargs['batch_size']
        f1          = rng.uniform(f_min, f_max / 2, batch_size)
        f2          = rng.uniform(f_max / 2, f_max, batch_size)
 
        fix_dur     = int(rng.uniform(300, 600) / dt)
        cue_dur     = int(rng.uniform(250, 350) / dt)
        phase1_dur  = int(rng.uniform(1800, 2600) / dt)
        switch_dur  = max(1, int(5 / dt))
        phase2_dur  = int(rng.uniform(1800, 2600) / dt)
 
    elif mode == 'test':
        batch_size  = 4
        f1          = np.full(batch_size, 0.03)
        f2          = np.full(batch_size, 0.07)
 
        fix_dur     = int(500 / dt)
        cue_dur     = int(300 / dt)
        phase1_dur  = int(2000 / dt)
        switch_dur  = max(1, int(5 / dt))
        phase2_dur  = int(2000 / dt)
 
    else:
        raise ValueError('Unknown mode: ' + str(mode))
 
    cue_on      = fix_dur
    cue_off     = cue_on + cue_dur
    phase1_on   = cue_off
    phase1_off  = phase1_on + phase1_dur
    switch_on   = phase1_off
    switch_off  = switch_on + switch_dur
    phase2_on   = switch_off
    phase2_off  = phase2_on + phase2_dur
    tdim        = phase2_off
 
    trial = Trial(config, tdim, batch_size)
    # Fixation drops at start of phase 1 (no traditional go cue).
    trial.add('fix_in', offs=cue_off)
    trial.add('fix_out', offs=cue_off)
 
    for i in range(batch_size):
        # Frequency cues during CUE_1.
        trial.x[cue_on:cue_off, i, SCALAR_IN_A] = f1[i] / f_max
        trial.x[cue_on:cue_off, i, SCALAR_IN_B] = f2[i] / f_max
 
        # Switch pulse on SCALAR_IN_B.
        trial.x[switch_on:switch_off, i, SCALAR_IN_B] = 1.0
 
        # Target: phase 1 sinusoid then phase 2 sinusoid.
        for t in range(phase1_on, phase1_off):
            trial.y[t, i, SCALAR_OUT_A] = np.sin(
                2 * np.pi * f1[i] * (t - phase1_on))
        for t in range(phase2_on, phase2_off):
            trial.y[t, i, SCALAR_OUT_A] = np.sin(
                2 * np.pi * f2[i] * (t - phase2_on))
 
    check_ons = phase1_on + int(100 / dt)
    trial.add_c_mask(pre_offs=phase1_on, post_ons=check_ons)
 
    trial.epochs = {
        'fix1'   : (None, cue_on),
        'cue_1'  : (cue_on, cue_off),
        'phase_1': (phase1_on, phase1_off),
        'switch' : (switch_on, switch_off),
        'phase_2': (phase2_on, phase2_off),
    }
 
    trial.f1 = f1
    trial.f2 = f2
    return trial
 
 
# ---------------------------------------------------------------------------
# T29: DelayedAssociation  (Compositional: binding + interference resistance)
# ---------------------------------------------------------------------------
 
def delayedassociation(config, mode, **kwargs):
    """
    T29: DelayedAssociation.
 
    See one (cue, angle) pair. Then a long stretch of distractor angles
    on modality 1 (no cues). Then the cue is shown alone. Recall the
    original angle.
 
    Tests interference resistance for within-trial associations.
 
    Cue is encoded as a 4-dim vector packed into ch 1-4 directly
    (same approach as T25). Angle is shown on modality 2 (ch 3-4) but
    during PAIR epoch we write directly since modality 2 overlaps ch 3-4.
 
    Simplified: cue is a discrete angle on modality 1, paired angle on
    modality 2, to avoid the n_input expansion issue.
 
    Inputs:  fixation (ch 0), cue angle on modality 1 (ch 1-2),
             paired angle on modality 2 (ch 3-4),
             distractor angles on modality 1 (ch 1-2) during DISTRACT
    Outputs: fixation (ch 0), recalled paired angle (ch 1-2)
    """
    dt     = config['dt']
    rng    = config['rng']
    n_cues = kwargs.get('n_cues', 8)
 
    cue_locs_all = np.linspace(0, 2 * np.pi, n_cues, endpoint=False)
 
    if mode == 'random':
        batch_size    = kwargs['batch_size']
        cue_ids       = rng.choice(n_cues, size=batch_size)
        cue_locs      = cue_locs_all[cue_ids]
        paired_angles = rng.uniform(0, 2 * np.pi, batch_size)
 
        fix_dur       = int(rng.uniform(300, 600) / dt)
        pair_dur      = int(rng.uniform(400, 800) / dt)
        distract_dur  = int(rng.uniform(2500, 4000) / dt)
        probe_dur     = int(rng.uniform(400, 600) / dt)
        resp_dur      = int(rng.uniform(300, 700) / dt)
        n_distractors = rng.randint(4, 10)
 
    elif mode == 'test':
        batch_size    = n_cues
        cue_ids       = np.arange(n_cues)
        cue_locs      = cue_locs_all
        paired_angles = (cue_locs + np.pi / 3) % (2 * np.pi)
 
        fix_dur       = int(500 / dt)
        pair_dur      = int(500 / dt)
        distract_dur  = int(3000 / dt)
        probe_dur     = int(500 / dt)
        resp_dur      = int(500 / dt)
        n_distractors = 6
 
    else:
        raise ValueError('Unknown mode: ' + str(mode))
 
    pair_on      = fix_dur
    pair_off     = pair_on + pair_dur
    distract_on  = pair_off
    distract_off = distract_on + distract_dur
    probe_on     = distract_off
    probe_off    = probe_on + probe_dur
    fix_offs     = probe_off
    tdim         = fix_offs + resp_dur
 
    trial = Trial(config, tdim, batch_size)
    trial.add('fix_in', offs=fix_offs)
    trial.add('fix_out', offs=fix_offs)
 
    # PAIR: cue on modality 1, paired angle on modality 2.
    trial.add('stim', cue_locs,      ons=pair_on, offs=pair_off, mods=1)
    trial.add('stim', paired_angles, ons=pair_on, offs=pair_off, mods=2)
 
    # DISTRACT: random angles on modality 1 at irregular times.
    distract_spacing = distract_dur // (n_distractors + 1)
    for j in range(n_distractors):
        d_on  = distract_on + j * distract_spacing + rng.randint(0, distract_spacing // 3)
        d_dur = int(rng.uniform(200, 400) / dt)
        d_off = min(d_on + d_dur, distract_off - 5)
        d_locs = rng.uniform(0, 2 * np.pi, batch_size)
        trial.add('stim', d_locs, ons=d_on, offs=d_off, mods=1)
 
    # PROBE: cue alone on modality 1.
    trial.add('stim', cue_locs, ons=probe_on, offs=probe_off, mods=1)
 
    # Response: original paired angle.
    trial.add('out', paired_angles, ons=fix_offs)
 
    check_ons = fix_offs + int(100 / dt)
    trial.add_c_mask(pre_offs=fix_offs, post_ons=check_ons)
 
    trial.epochs = {
        'fix1'    : (None, pair_on),
        'pair'    : (pair_on, pair_off),
        'distract': (distract_on, distract_off),
        'probe'   : (probe_on, probe_off),
        'go1'     : (fix_offs, None),
    }
 
    trial.cue_ids      = cue_ids
    trial.cue_locs     = cue_locs
    trial.paired_angles = paired_angles
    return trial
 
 
# ---------------------------------------------------------------------------
# T30: SequentialDecision  (Compositional: evidence integration x3)
# ---------------------------------------------------------------------------
 
def sequentialdecision(config, mode, **kwargs):
    """
    T30: SequentialDecision.
 
    Three sequential decision blocks. In each block, two stimuli appear
    on modalities 1 and 2; the stronger one wins. The final response is
    the circular mean of the three winning angles — a composite of all
    three decisions.
 
    Inputs:  fixation (ch 0), stim A on modality 1 (ch 1-2),
             stim B on modality 2 (ch 3-4) — three blocks sequentially
    Outputs: fixation (ch 0), composite response angle (ch 1-2)
    """
    dt  = config['dt']
    rng = config['rng']
    n_blocks = 3
 
    if mode == 'random':
        batch_size  = kwargs['batch_size']
        block_dur   = int(rng.uniform(900, 1300) / dt)
        fix_dur     = int(rng.uniform(300, 700) / dt)
        resp_dur    = int(rng.uniform(300, 700) / dt)
 
    elif mode == 'test':
        batch_size  = 8
        block_dur   = int(1000 / dt)
        fix_dur     = int(500 / dt)
        resp_dur    = int(500 / dt)
 
    else:
        raise ValueError('Unknown mode: ' + str(mode))
 
    fix_offs = fix_dur + n_blocks * block_dur
    tdim     = fix_offs + resp_dur
 
    trial = Trial(config, tdim, batch_size)
    trial.add('fix_in', offs=fix_offs)
    trial.add('fix_out', offs=fix_offs)
 
    # Generate stimuli for each block and compute composite response.
    # Composite = circular mean of winning angles across blocks.
    sin_sum = np.zeros(batch_size)
    cos_sum = np.zeros(batch_size)
 
    for blk in range(n_blocks):
        blk_on  = fix_dur + blk * block_dur
        blk_off = blk_on + block_dur
 
        theta_A = rng.uniform(0, 2 * np.pi, batch_size)
        theta_B = rng.uniform(0, 2 * np.pi, batch_size)
        amp_A   = rng.uniform(0.3, 1.0, batch_size)
        amp_B   = rng.uniform(0.3, 1.0, batch_size)
 
        # Show stimuli on modalities 1 and 2.
        trial.add('stim', theta_A, ons=blk_on, offs=blk_off,
                  strengths=amp_A, mods=1)
        trial.add('stim', theta_B, ons=blk_on, offs=blk_off,
                  strengths=amp_B, mods=2)
 
        # Winner of this block.
        winner = np.where(amp_A > amp_B, theta_A, theta_B)
        sin_sum += np.sin(winner)
        cos_sum += np.cos(winner)
 
    # Circular mean of winners.
    composite = np.arctan2(sin_sum, cos_sum) % (2 * np.pi)
 
    trial.add('out', composite, ons=fix_offs)
 
    check_ons = fix_offs + int(100 / dt)
    trial.add_c_mask(pre_offs=fix_offs, post_ons=check_ons)
 
    ep = {'fix1': (None, fix_dur)}
    for blk in range(n_blocks):
        ep[f'block_{blk+1}'] = (fix_dur + blk * block_dur,
                                 fix_dur + (blk + 1) * block_dur)
    ep['go1'] = (fix_offs, None)
    trial.epochs = ep
 
    trial.composite = composite
    return trial
 
# ---------------------------------------------------------------------------
# Complete rule registration for all 30 tasks
# ---------------------------------------------------------------------------
# Paste this into task.py after all function definitions.
# Functions from new_tasks_prototype.py and new_tasks_prototype_part2.py
# must be imported or copied into task.py first.

rules_dict['all_30'] = [
    # --- Yang/Driscoll (T01-T10): existing functions ---
    'fdgo',             # T01 DelayPro
    'fdanti',           # T02 DelayAnti
    'delaygo',          # T03 MemoryPro
    'delayanti',        # T04 MemoryAnti
    'delaydm1',         # T05 DM
    'delaydmanti',      # T06 DMAnti          ** new wrapper needed (see below)
    'contextdelaydm1',  # T07 ContextDM-A
    'contextdelaydm2',  # T08 ContextDM-B
    'dmsgo',            # T09 DelayMatchSample
    'dmsnogo',          # T10 DelayNonMatchSample

    # --- WM probes (T11-T12) ---
    'extendedmemory',   # T11 ExtendedMemory  ** new wrapper needed (see below)
    'multiitemrecall',  # T12 MultiItemRecall ** NOT YET WRITTEN (see note)

    # --- Counting/Timing (T13-T15) ---
    'pulsecounting',        # T13
    'intervalreproduction', # T14
    'pulserateestimation',  # T15

    # --- Sequence/Rhythm (T16-T17) ---
    'rhythmgeneration', # T16
    'sequencerecall',   # T17

    # --- Flip-flop (T18-T19) ---
    'toggle',           # T18
    'conditionaltoggle',# T19

    # --- Associative (T20-T22) ---
    'cueassoc',         # T20
    'pairedassoc',      # T21
    'reversallearning', # T22

    # --- Online learning (T23-T25) ---
    'onlinelinearreg',    # T23
    'onlinenonlinearreg', # T24
    'fewshotclassif',     # T25

    # --- Compositional (T26-T30) ---
    'memorydm',            # T26
    'countandrecall',      # T27
    'conditionalrhythm',   # T28
    'delayedassociation',  # T29
    'sequentialdecision',  # T30
]

# Smaller subsets for staged training
rules_dict['driscoll_15'] = [
    'fdgo', 'reactgo', 'delaygo', 'fdanti', 'reactanti', 'delayanti',
    'delaydm1', 'delaydm2', 'contextdelaydm1', 'contextdelaydm2',
    'multidelaydm', 'dmsgo', 'dmsnogo', 'dmcgo', 'dmcnogo',
]

rules_dict['new_tasks'] = [
    'pulsecounting', 'intervalreproduction', 'pulserateestimation',
    'rhythmgeneration', 'sequencerecall', 'toggle', 'conditionaltoggle',
    'cueassoc', 'pairedassoc', 'reversallearning',
    'onlinelinearreg', 'onlinenonlinearreg', 'fewshotclassif',
    'memorydm', 'countandrecall', 'conditionalrhythm',
    'delayedassociation', 'sequentialdecision',
]

rules_dict['compositional'] = [
    'memorydm', 'countandrecall', 'conditionalrhythm',
    'delayedassociation', 'sequentialdecision',
]

rules_dict['across_trial'] = [
    'cueassoc', 'reversallearning',
]


# ---------------------------------------------------------------------------
# rule_mapping additions
# ---------------------------------------------------------------------------
# Add these to the existing rule_mapping dict.

rule_mapping.update({
    # T06 wrapper -- respond to the WEAKER stimulus
    'delaydmanti'       : lambda c, m, **kw: _delaydm_anti(c, m, **kw),

    # T11 wrapper -- same as delaygo_ but longer delay
    'extendedmemory'    : lambda c, m, **kw: delaygo_(c, m, False,
                              delay_range=(5000, 10000), **kw),

    # T12 -- NOT YET WRITTEN, placeholder raises clearly
    'multiitemrecall'   : lambda c, m, **kw: (_ for _ in ()).throw(
                              NotImplementedError('T12 multiitemrecall not yet implemented')),

    # New tasks from prototype files
    'pulsecounting'       : pulsecounting,
    'intervalreproduction': intervalreproduction,
    'pulserateestimation' : pulserateestimation,
    'rhythmgeneration'    : rhythmgeneration,
    'sequencerecall'      : sequencerecall,
    'toggle'              : toggle,
    'conditionaltoggle'   : conditionaltoggle,
    'cueassoc'            : cueassoc,
    'pairedassoc'         : pairedassoc,
    'reversallearning'    : reversallearning,
    'onlinelinearreg'     : onlinelinearreg,
    'onlinenonlinearreg'  : onlinenonlinearreg,
    'fewshotclassif'      : fewshotclassif,
    'memorydm'            : memorydm,
    'countandrecall'      : countandrecall,
    'conditionalrhythm'   : conditionalrhythm,
    'delayedassociation'  : delayedassociation,
    'sequentialdecision'  : sequentialdecision,
})


# ---------------------------------------------------------------------------
# rule_name additions
# ---------------------------------------------------------------------------

rule_name.update({
    'delaydmanti'         : 'Dly DM Anti',
    'extendedmemory'      : 'Ext Memory',
    'multiitemrecall'     : 'Multi-Item Recall',
    'pulsecounting'       : 'Pulse Count',
    'intervalreproduction': 'Interval Repro',
    'pulserateestimation' : 'Rate Estim',
    'rhythmgeneration'    : 'Rhythm Gen',
    'sequencerecall'      : 'Seq Recall',
    'toggle'              : 'Toggle',
    'conditionaltoggle'   : 'Cond Toggle',
    'cueassoc'            : 'Cue Assoc',
    'pairedassoc'         : 'Paired Assoc',
    'reversallearning'    : 'Reversal Learn',
    'onlinelinearreg'     : 'Online Lin Reg',
    'onlinenonlinearreg'  : 'Online Nonlin Reg',
    'fewshotclassif'      : 'Few-Shot Classif',
    'memorydm'            : 'Memory DM',
    'countandrecall'      : 'Count & Recall',
    'conditionalrhythm'   : 'Cond Rhythm',
    'delayedassociation'  : 'Delayed Assoc',
    'sequentialdecision'  : 'Sequential Dec',
})


# ---------------------------------------------------------------------------
# Two small wrapper functions needed in task.py
# ---------------------------------------------------------------------------

def _delaydm_anti(config, mode, **kwargs):
    """T06 DMAnti: like delaydm but respond to the WEAKER stimulus."""
    # Generate a normal delaydm trial then flip the output locs.
    trial = _delaydm(config, mode, stim_mod=1, **kwargs)
    # The output was set to the stronger stim location. Flip by pi.
    # Find timesteps where output is nonzero and rotate by pi.
    nonzero = np.any(trial.y[:, :, 1:] != 0, axis=2)  # (T, B)
    for i in range(trial.batch_size):
        resp_ts = np.where(nonzero[:, i])[0]
        if len(resp_ts):
            t0 = resp_ts[0]
            # Current response angle from sin/cos.
            s = trial.y[t0, i, 1]
            c = trial.y[t0, i, 2]
            angle = np.arctan2(s, c)
            anti  = angle + np.pi
            trial.y[t0:, i, 1] = np.sin(anti)
            trial.y[t0:, i, 2] = np.cos(anti)
    return trial


def extendedmemory(config, mode, **kwargs):
    """T11 ExtendedMemory: delaygo with 5-10 second delay instead of 200-1600ms."""
    # Temporarily patch the delay range by overriding rng behavior.
    # Simplest approach: call delaygo_ and accept it uses its internal ranges,
    # then override. Better: add delay_range kwarg to delaygo_ when integrating.
    # For now, use a fresh Trial with the long delay directly.
    dt  = config['dt']
    rng = config['rng']

    if mode == 'random':
        batch_size = kwargs['batch_size']
        stim_locs  = rng.rand(batch_size) * 2 * np.pi
        stim_mod   = rng.choice([1, 2])
        stim_ons   = int(rng.uniform(300, 700) / dt)
        stim_offs  = stim_ons + int(rng.uniform(200, 1600) / dt)
        # Long delay: 5000-10000 ms
        fix_offs   = stim_offs + int(rng.uniform(5000, 10000) / dt)
        tdim       = fix_offs + int(rng.uniform(300, 700) / dt)
    elif mode == 'test':
        batch_size = 40
        stim_locs  = np.linspace(0, 2 * np.pi, batch_size, endpoint=False)
        stim_mod   = 1
        stim_ons   = int(500 / dt)
        stim_offs  = int(1000 / dt)
        fix_offs   = int(8000 / dt)
        tdim       = int(8500 / dt)
    else:
        raise ValueError('Unknown mode: ' + str(mode))

    check_ons = fix_offs + int(100 / dt)
    trial = Trial(config, tdim, batch_size)
    trial.add('fix_in', offs=fix_offs)
    trial.add('stim', stim_locs, ons=stim_ons, offs=stim_offs, mods=stim_mod)
    trial.add('fix_out', offs=fix_offs)
    trial.add('out', stim_locs, ons=fix_offs)
    trial.add_c_mask(pre_offs=fix_offs, post_ons=check_ons)
    trial.epochs = {
        'fix1'      : (None, stim_ons),
        'stim1'     : (stim_ons, stim_offs),
        'long_delay': (stim_offs, fix_offs),
        'go1'       : (fix_offs, None),
    }
    return trial
