# June 11, 2026

## Goal
Get the Driscoll flexible_multitask code running on Kempner.

## Repository Structure
stepnet/        # train from scratch
analysis/       # reproduce paper figures
transfer_learn/ # later, pretrained/continual experiments
utils/          # helper code

## Questions

1. Where are the 15 task definitions?
2. Where is the network architecture?
3. Where is the training script?
4. Where are pretrained networks stored?

## Progress

- Connected to FASRC through VS Code Remote SSH
- Cloned flexible_multitask repository

## Next Steps

- Read README
- Locate task generation code

## Repo structure from README

- `stepnet/`: training networks from random initialization
- `transfer_learn/`: training networks starting from pretrained networks
- `utils/`: shared utility functions
- `analysis/`: reproduces paper figures using open-source data

## Current goal

Find:
- task definitions
- RNN/model code
- training script
- analysis scripts

# Training Pipeline

1. Task generation:
   ?

2. Trial generation:
   ?

3. Network architecture:
   ?

4. Training loop:
   

5. Checkpoint saving:
   ?

6. Analysis:
   ?

# June 15, 2026

# Training Driscoll
- on hold until cluster is running again
- need to figure out my username

# Task Setup
- create and implement code for new tasks for meta-learning

# Background Reading
- re-read and understand Driscoll, Yang
- look into meta-learning frameworks and possible ways to design them

# June 17, 2026

# Transformer
Question:
Can a strong generic sequence model solve this task?

If transformer succeeds and RNN fails:
    task may require better memory/in-context learning.

If plastic RNN succeeds where standard RNN fails:
    evidence that plasticity helps.

If transformer and plastic RNN both succeed:
    compare mechanisms.

## Smoke test result

Date: 2026-06-19  
Node: A100 compute node, ran on CPU due CUDA library issue  
Script: stepnet/general_model_train_smoke.py  
Tasks: delaygo, delayanti  
n_rnn: 32  
Result: model built, trained, saved checkpoint, optimization finished  
Conclusion: code pipeline works end-to-end

## 2026-06-22 — First successful 15-task GPU run

Compute:
- Partition: kempner
- GPU: A100 40GB
- Node: holygpu8a19204

Environment:
- python/3.10.13-fasrc01
- cuda/11.8.0-fasrc01
- cudnn/8.9.2.26_cuda11-fasrc01
- TensorFlow 2.13.1
- GPU detected successfully

Training:
- Script: general_model_train_15task_test.py
- n_rnn = 128
- ruleset = all (15 tasks)
- max_steps = 96000
- display_step = ...

Results:
- fdgo = 0.68
- reactgo = 0.40
- fdanti = 0.64
- reactanti = 0.53
- harder memory/decision tasks still low

Conclusion:
- Full Driscoll pipeline runs on GPU.
- Checkpoint saving works.
- Need longer runs for convergence.

## 2026-06-23 — 5M-trial Driscoll 15-task run

Run completed successfully.

Config:
- Original 15 tasks
- LeakyRNN
- n_rnn = 128
- max_steps = 5,000,000
- Final printed trial: 4,992,000
- Runtime: 10,704 s ≈ 2.97 hours
- Output: data/driscoll_15task_test/0

Final performance:
- fdgo: 0.99
- reactgo: 0.99
- delaygo: 0.82
- fdanti: 0.99
- reactanti: 0.99
- delayanti: 0.73
- delaydm1: 0.66
- delaydm2: 0.73
- contextdelaydm1: 0.68
- contextdelaydm2: 0.68
- multidelaydm: 0.77
- dmsgo: 0.72
- dmsnogo: 0.64
- dmcgo: 0.87
- dmcnogo: 0.89

Conclusion:
The original Driscoll 15-task training pipeline runs successfully on the Kempner A100 partition. Most tasks substantially improved by 5M trials, though several memory/decision tasks are not fully converged yet.


## 2026-06-24 - 20M Driscoll 15-task run

## Configuration

* Model: LeakyRNN
* Hidden units: 128
* Tasks: Original 15-task Driscoll suite
* Training trials: 20,000,000
* GPU: A100 40GB
* Output directory: `data/driscoll_15task_20M/0`
* Runtime: 19,008.6 seconds (~5.3 hours)

## Final Performance

| Task            | Performance |
| --------------- | ----------: |
| fdgo            |        0.99 |
| reactgo         |        0.99 |
| delaygo         |        0.86 |
| fdanti          |        0.99 |
| reactanti       |        0.99 |
| delayanti       |        0.88 |
| delaydm1        |        0.74 |
| delaydm2        |        0.79 |
| contextdelaydm1 |        0.73 |
| contextdelaydm2 |        0.78 |
| multidelaydm    |        0.80 |
| dmsgo           |        0.78 |
| dmsnogo         |        0.86 |
| dmcgo           |        0.92 |
| dmcnogo         |        0.91 |

## Comparison to 5M-Trial Run

Substantial improvements observed on:

* delaygo
* delayanti
* delaydm1
* delaydm2
* contextdelaydm1
* contextdelaydm2
* multidelaydm
* dmsgo
* dmsnogo
* dmcgo
* dmcnogo

The network was not fully converged at 5M trials.

## Conclusions

* GPU training is functioning correctly.
* Original Driscoll training pipeline successfully reproduces expected learning behavior.
* Performance continues improving substantially between 5M and 20M trials.
* A longer run (e.g., 100M trials) is justified if the goal is full reproduction.

## Next Steps

1. Launch a longer reproduction run (100M trials).
2. Preserve this trained checkpoint for later analysis.
3. Begin integrating T20 (CueAssoc).
4. Begin integrating T21 (PairedAssociation).
5. Compare learning dynamics of new tasks against the original 15-task suite.
