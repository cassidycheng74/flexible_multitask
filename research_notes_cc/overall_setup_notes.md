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