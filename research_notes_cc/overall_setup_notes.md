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

## 2026-06-24
Added all new tasks except T12 to the task_new_drafts code, need to implement into original Driscoll structure before running/training standard RNN on them

## 2026-06-26
## Phase 1 Results: Driscoll Reproduction

### Model
- Architecture: LeakyRNN, softplus, diag init, 128 units
- Tasks: 15 (full Driscoll ruleset)
- Training: max_steps=1e8, lr=1e-3, seed=0
- Path: data/all/LeakyRNN/softplus/diag/15_tasks/128_n_rnn/lr6l2_w6_h6_.../0

### Final Performance (all 15 tasks)
- fdgo: 0.997, reactgo: 0.990, delaygo: 0.970
- fdanti: 0.987, reactanti: 0.990, delayanti: 0.731
- delaydm1: 0.883, delaydm2: 0.731
- contextdelaydm1: 0.817, contextdelaydm2: 0.685
- multidelaydm: 0.770
- dmsgo: 0.937, dmsnogo: 0.637, dmcgo: 0.870, dmcnogo: 0.890

### Key Analyses
1. Training curves: all 15 tasks learned, correct difficulty ordering
2. DelayGo PCA: clear ring structure in PC1-PC2 space (PC1=27.6%, PC2=17.5%)
   - Endpoints ordered continuously by stimulus angle = ring attractor confirmed
3. Shared subspace: DelayAnti projects onto DelayGo PCA axes as mirror image
   - Pro tasks in upper PC2, anti tasks in lower PC2
   - Same ring manifold, different readout = shared dynamical motif confirmed

### Conclusion
Model successfully reproduces Driscoll et al. 2024 key findings.
Ring attractor dynamics confirmed for memory tasks.
Shared subspace confirmed for pro/anti task pairs.
Ready to proceed to Phase 2 (new task implementation).

## 2026-06-29: Driscoll Reproduction

Model: LeakyRNN, softplus activation, diagonal init, 128 units, 15 tasks, seed 0

Training: Driscoll's exact train.py, max_steps=1e8, lr=1e-3

Model path: data/all/LeakyRNN/softplus/diag/15_tasks/128_n_rnn/lr6l2_.../0

What was reproduced:
Training dynamics (fig_training_curves, fig_final_performance)

- All 15 tasks learned successfully
- Correct difficulty ordering confirmed, simple stimulus-response tasks (fdgo 0.997, fdanti 0.987) hit near-ceiling performance early, while working memory tasks (delaygo 0.970, delayanti 0.731) and context tasks (contextdelaydm1 0.817) took longer and reached lower asymptotes

# Ring attractor dynamics (fig_delaygo_pca)

Hidden states during the delaygo task form a low-dimensional ring manifold in PCA space. PC1 (27.6% variance) captures the fixation-to-response transition direction. PC2 and PC3 together encode stimulus angle as position on the ring, with colors ordered continuously around the endpoint cloud. This is the ring attractor signature Driscoll reports as the core memory mechanism — the network maintains stimulus angle as a stable position on a manifold rather than through sustained high firing.

# Different motifs for different task families (fig_delaydm1_pca)

Decision-making tasks produce qualitatively different dynamics — two clusters of endpoints rather than a ring, corresponding to the two possible decisions. This confirms that memory tasks and decision tasks use different dynamical solutions, which is the "multiple motifs" claim in the paper title.

# Shared subspace between task pairs (fig_shared_subspace)

When delayanti hidden states are projected onto the PCA axes defined by delaygo, they form a mirror image — endpoints in the lower PC2 region with the same continuous color ordering as delaygo but reflected. This confirms that pro and anti memory tasks use the same ring attractor manifold with different output mappings, not separate circuits. This is Driscoll's central shared subspace finding.

# Global task geometry (fig_all_tasks_pca)

All 15 tasks projected into a global PCA space show family-specific geometric structure — memory tasks show ring-like distributions, decision tasks show two-cluster structure, context tasks show more complex geometry. Tasks within the same family look similar; tasks across families look different.

# Unit selectivity (fig_variance_matrix)

The variance matrix shows a sparse population of active units (roughly 30 out of 128) with the remaining units contributing little variance across any task. Active units are broadly tuned across multiple tasks rather than strictly task-specific. The sparsity pattern matches Driscoll's finding, though the block structure is less clean than her 256-unit network — expected at half the network size. Averaging across time epochs blurs the epoch-specific selectivity Driscoll shows in her version.

# Quantitative shared subspace (fig_cross_task_variance)

The 15×15 cross-task variance matrix shows clear block structure — tasks within the same computational family (go, anti, DM, context DM, match, category-match) explain high fractions of each other's neural variance. Tasks across families explain near-zero variance of each other. The go and anti families show moderate cross-block values, confirming they share the ring geometry. This is the quantitative version of the shared subspace result and directly replicates Driscoll's Fig 4d/g finding.

# Context period structure (fig_context_period_pca)

Hidden states during the fixation period, before any stimulus arrives, already show task-family clustering in PCA space. The network has committed to a computational strategy based on the rule input alone. This matches Driscoll's Fig 4a result.

# Did Not Reproduce:
- fixed point analysis
- detailed variance matrix

# Overall conclusion
The key scientific claims of Driscoll et al. 2024 are confirmed at the representational level:

- Ring attractor dynamics in memory tasks
- Distinct motifs for distinct task families
- Shared neural subspace within task families, distinct subspaces across families
- Context period already encodes computational strategy before stimulus onset