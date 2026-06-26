"""
generate_activity.py

Generates and saves neural activity files needed by the Driscoll
analysis notebooks. Run this once after training before opening
any of the figure notebooks.

Produces for each mode (test, random):
    state.npz        - hidden states (T, B, n_rnn) per task
    trial_input.npz  - inputs (T, B, n_input) per task
    trial_output.npz - network outputs (T, B, n_output) per task
    model_hparams.npz - hyperparameters
    model_params.npz  - trained weight matrices

Usage:
    python generate_activity.py

Output will be saved to:
    data/driscoll_15task_test/activity/<seed>/<mode>/
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import numpy as np

# Point to the stepnet source directory.
STEPNET_DIR = os.path.join(os.path.dirname(__file__), '..', 'stepnet')
sys.path.insert(0, STEPNET_DIR)

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from task import generate_trials, rules_dict
from network import Model
import tools

# ---------------------------------------------------------------------------
# Configuration — edit these if your paths differ
# ---------------------------------------------------------------------------

# Directory where your trained model lives.
MODEL_DIR = 'data/all/LeakyRNN/softplus/diag/15_tasks/128_n_rnn/lr6l2_w6_h6_sig_rec0.05_sig_x0.1_w_rec_coeff1_fdgo_reactgo_delaygo_fdanti_reactanti_delayanti_delaydm1_delaydm2_contextdelaydm1_contextdelaydm2_multidelaydm_dmsgo_dmsnogo_dmcgo_dmcnogo/0'

# Where to save the activity files.
# Will create subdirectories: <save_root>/<seed>/<mode>/

SAVE_ROOT = 'data/all/LeakyRNN/softplus/diag/15_tasks/128_n_rnn/lr6l2_w6_h6_sig_rec0.05_sig_x0.1_w_rec_coeff1_fdgo_reactgo_delaygo_fdanti_reactanti_delayanti_delaydm1_delaydm2_contextdelaydm1_contextdelaydm2_multidelaydm_dmsgo_dmsnogo_dmcgo_dmcnogo/0/activity'

# Seeds to process. You only trained seed 0, so just [0].
SEEDS = [0]

# Tasks to extract activity for.
RULE_TRAINS = [
    'fdgo', 'reactgo', 'delaygo',
    'fdanti', 'reactanti', 'delayanti',
    'delaydm1', 'delaydm2',
    'contextdelaydm1', 'contextdelaydm2', 'multidelaydm',
    'dmsgo', 'dmsnogo', 'dmcgo', 'dmcnogo',
]

# Number of trials to generate per task.
BATCH_SIZE = 500

# ---------------------------------------------------------------------------
# Main generation loop
# ---------------------------------------------------------------------------

for seed in SEEDS:
    model_dir = MODEL_DIR

    for mode in ['test', 'random']:

        print(f'\n=== Seed {seed} | Mode {mode} ===')

        noise_on = (mode == 'random')

        trial_dict  = {}
        h_dict      = {}
        state_dict  = {}
        output_dict = {}
        input_dict  = {}

        model = Model(model_dir)

        with tf.Session() as sess:
            model.restore()
            # Zero out recurrent noise during activity extraction.
            model._sigma = 0

            hparams    = model.hp
            var_list   = model.var_list
            params     = [sess.run(var) for var in var_list]

            for rule in RULE_TRAINS:
                print(f'  Generating trials for {rule}...', end=' ')

                trial = generate_trials(
                    rule, hparams,
                    mode=mode,
                    noise_on=noise_on,
                    batch_size=BATCH_SIZE,
                )

                feed_dict = tools.gen_feed_dict(model, trial, hparams)
                h = sess.run(model.h, feed_dict=feed_dict)

                state_dict[rule]  = h
                output_dict[rule] = trial.y
                input_dict[rule]  = trial.x

                print(f'done. state shape: {h.shape}')

        # Save all files for this seed/mode combination.
        save_dir = os.path.join(SAVE_ROOT, str(seed), mode)
        os.makedirs(save_dir, exist_ok=True)

        print(f'\nSaving to {save_dir}...')

        np.savez(os.path.join(save_dir, 'state.npz'),        **state_dict)
        np.savez(os.path.join(save_dir, 'trial_input.npz'),  **input_dict)
        np.savez(os.path.join(save_dir, 'trial_output.npz'), **output_dict)
        np.savez(os.path.join(save_dir, 'model_hparams.npz'),**hparams)
        np.savez(os.path.join(save_dir, 'model_params.npz'),
         **{f'param_{i}': p for i, p in enumerate(params)})

        print(f'Saved: state, trial_input, trial_output, model_hparams, model_params')

print('\nDone. Activity files saved to:', SAVE_ROOT)
print('\nNext step: open the figure notebooks and update save_dir to:')
for seed in SEEDS:
    for mode in ['test', 'random']:
        print(f'  {os.path.join(SAVE_ROOT, str(seed), mode)}')