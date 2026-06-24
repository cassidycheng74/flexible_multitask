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
* Most tasks >0.8 performance.
* Sensorimotor tasks ~0.99.
* Harder memory/decision tasks still improving.

## Next Steps

1. Launch a longer reproduction run (100M trials).
2. Preserve this trained checkpoint for later analysis.
3. Begin integrating T20 (CueAssoc).
4. Begin integrating T21 (PairedAssociation).
5. Compare learning dynamics of new tasks against the original 15-task suite.
