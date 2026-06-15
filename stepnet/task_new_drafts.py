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


def cueassoc(config, mode, **kwargs):
    """
    T20 prototype: CueResponseAssoc.

    Trial structure:
        FIX -> CUE -> DELAY -> GO

    Current simplified version:
        - cue identity is represented as a discrete angle on modality 1
        - each cue maps to a target response angle
        - mapping is deterministic and fixed, so this is supervised learning,
          not full across-lifetime meta-learning yet

    Inputs:
        x[:, :, 0] = fixation input
        x[:, :, 1:3] = cue angle, encoded as sin/cos on modality 1

    Outputs:
        y[:, :, 0] = fixation output
        y[:, :, 1:3] = associated response angle
    """
    dt = config['dt']
    rng = config['rng']

    n_cues = kwargs.get('n_cues', 8)

    # Cue identities are discrete angles.
    cue_locs_all = np.linspace(0, 2 * np.pi, n_cues, endpoint=False)

    # First simple mapping: each cue maps to an angle rotated by 90 degrees.
    # You can later replace this with random lifetime-specific mappings.
    response_locs_all = (cue_locs_all + np.pi / 2) % (2 * np.pi)

    if mode == 'random':
        batch_size = kwargs['batch_size']

        cue_ids = rng.choice(n_cues, size=batch_size)
        cue_locs = cue_locs_all[cue_ids]
        response_locs = response_locs_all[cue_ids]

        cue_ons = int(rng.uniform(300, 700) / dt)
        cue_offs = cue_ons + int(rng.uniform(300, 800) / dt)
        fix_offs = cue_offs + int(rng.uniform(200, 800) / dt)
        tdim = fix_offs + int(rng.uniform(300, 700) / dt)

    elif mode == 'test':
        batch_size = n_cues

        cue_ids = np.arange(n_cues)
        cue_locs = cue_locs_all
        response_locs = response_locs_all

        cue_ons = int(500 / dt)
        cue_offs = int(1000 / dt)
        fix_offs = int(1500 / dt)
        tdim = int(2000 / dt)

    elif mode == 'psychometric':
        p = kwargs['params']

        cue_ids = np.array(p['cue_ids'])
        batch_size = len(cue_ids)

        cue_locs = cue_locs_all[cue_ids]
        response_locs = response_locs_all[cue_ids]

        cue_ons = int(p.get('cue_ons', 500) / dt)
        cue_offs = int(p.get('cue_offs', 1000) / dt)
        fix_offs = int(p.get('fix_offs', 1500) / dt)
        tdim = int(p.get('tdim', 2000) / dt)

    else:
        raise ValueError('Unknown mode: ' + str(mode))

    check_ons = fix_offs + int(100 / dt)

    trial = Trial(config, tdim, batch_size)

    # Fixation input stays on until response.
    trial.add('fix_in', offs=fix_offs)

    # Cue is represented as a circular stimulus on modality 1.
    trial.add('stim', cue_locs, ons=cue_ons, offs=cue_offs, mods=1)

    # Fixation output should be high before go cue.
    trial.add('fix_out', offs=fix_offs)

    # After go cue, output the associated response angle.
    trial.add('out', response_locs, ons=fix_offs)

    trial.add_c_mask(pre_offs=fix_offs, post_ons=check_ons)

    trial.epochs = {
        'fix1': (None, cue_ons),
        'cue1': (cue_ons, cue_offs),
        'delay1': (cue_offs, fix_offs),
        'go1': (fix_offs, None),
    }

    # Extra metadata for debugging/analysis.
    trial.cue_ids = cue_ids
    trial.cue_locs = cue_locs
    trial.response_locs = response_locs

    return trial

def pairedassoc(config, mode, **kwargs):
    """
    T21 prototype: PairedAssociation.

    Within one trial:
        PAIR_A: cue A + angle A
        PAIR_B: cue B + angle B
        PROBE: one cue alone
        RESP: output the angle paired with the probed cue

    This tests within-trial associative binding.
    """
    dt = config['dt']
    rng = config['rng']

    n_cues = kwargs.get('n_cues', 8)
    cue_locs_all = np.linspace(0, 2 * np.pi, n_cues, endpoint=False)

    if mode == 'random':
        batch_size = kwargs['batch_size']

        cue_ids_a = rng.choice(n_cues, size=batch_size)
        cue_ids_b = rng.choice(n_cues, size=batch_size)

        # Avoid identical cue pairs when possible.
        same = cue_ids_b == cue_ids_a
        cue_ids_b[same] = (cue_ids_b[same] + 1) % n_cues

        cue_locs_a = cue_locs_all[cue_ids_a]
        cue_locs_b = cue_locs_all[cue_ids_b]

        angle_a = rng.uniform(0, 2 * np.pi, batch_size)
        angle_b = rng.uniform(0, 2 * np.pi, batch_size)

        probe_is_a = rng.choice([0, 1], size=batch_size).astype(bool)
        probe_locs = np.where(probe_is_a, cue_locs_a, cue_locs_b)
        response_locs = np.where(probe_is_a, angle_a, angle_b)

        fix_dur = int(rng.uniform(300, 700) / dt)
        pair_dur = int(rng.uniform(400, 800) / dt)
        delay1_dur = int(rng.uniform(300, 800) / dt)
        delay2_dur = int(rng.uniform(300, 800) / dt)
        probe_dur = int(rng.uniform(300, 600) / dt)
        resp_dur = int(rng.uniform(300, 700) / dt)

    elif mode == 'test':
        # Deterministic small test set.
        batch_size = n_cues

        cue_ids_a = np.arange(n_cues)
        cue_ids_b = (cue_ids_a + 1) % n_cues

        cue_locs_a = cue_locs_all[cue_ids_a]
        cue_locs_b = cue_locs_all[cue_ids_b]

        angle_a = (cue_locs_a + np.pi / 4) % (2 * np.pi)
        angle_b = (cue_locs_b + np.pi / 2) % (2 * np.pi)

        probe_is_a = np.arange(n_cues) % 2 == 0
        probe_locs = np.where(probe_is_a, cue_locs_a, cue_locs_b)
        response_locs = np.where(probe_is_a, angle_a, angle_b)

        fix_dur = int(500 / dt)
        pair_dur = int(500 / dt)
        delay1_dur = int(500 / dt)
        delay2_dur = int(500 / dt)
        probe_dur = int(500 / dt)
        resp_dur = int(500 / dt)

    else:
        raise ValueError('Unknown mode: ' + str(mode))

    fix_on = 0
    pair_a_on = fix_dur
    pair_a_off = pair_a_on + pair_dur
    pair_b_on = pair_a_off + delay1_dur
    pair_b_off = pair_b_on + pair_dur
    probe_on = pair_b_off + delay2_dur
    probe_off = probe_on + probe_dur
    fix_offs = probe_off
    tdim = fix_offs + resp_dur

    check_ons = fix_offs + int(100 / dt)

    trial = Trial(config, tdim, batch_size)

    trial.add('fix_in', offs=fix_offs)

    # PAIR_A: cue A on modality 1, angle A on modality 2.
    trial.add('stim', cue_locs_a, ons=pair_a_on, offs=pair_a_off, mods=1)
    trial.add('stim', angle_a, ons=pair_a_on, offs=pair_a_off, mods=2)

    # PAIR_B: cue B on modality 1, angle B on modality 2.
    trial.add('stim', cue_locs_b, ons=pair_b_on, offs=pair_b_off, mods=1)
    trial.add('stim', angle_b, ons=pair_b_on, offs=pair_b_off, mods=2)

    # PROBE: cue alone on modality 1.
    trial.add('stim', probe_locs, ons=probe_on, offs=probe_off, mods=1)

    trial.add('fix_out', offs=fix_offs)
    trial.add('out', response_locs, ons=fix_offs)

    trial.add_c_mask(pre_offs=fix_offs, post_ons=check_ons)

    trial.epochs = {
        'fix1': (None, pair_a_on),
        'pair_a': (pair_a_on, pair_a_off),
        'delay1': (pair_a_off, pair_b_on),
        'pair_b': (pair_b_on, pair_b_off),
        'delay2': (pair_b_off, probe_on),
        'probe': (probe_on, probe_off),
        'go1': (fix_offs, None),
    }

    trial.cue_ids_a = cue_ids_a
    trial.cue_ids_b = cue_ids_b
    trial.cue_locs_a = cue_locs_a
    trial.cue_locs_b = cue_locs_b
    trial.angle_a = angle_a
    trial.angle_b = angle_b
    trial.probe_is_a = probe_is_a
    trial.response_locs = response_locs

    return trial