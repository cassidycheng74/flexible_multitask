# Future Meta-Plasticity Versions

## Why this document exists

The current task implementations are designed to fit the Driscoll/Yang multitask framework.

These tasks can likely be solved by standard backpropagation-trained recurrent networks with fixed weights.

The long-term goal of the project is different:

* local learning rules
* fast synaptic plasticity
* continual adaptation
* learning within a lifetime

This document describes how each prototype task can be transformed into a true meta-learning benchmark.

---

# T20: CueAssoc

## Current Version

Cue identities map to fixed response angles.

Example:

cue_A → 30°
cue_B → 120°
cue_C → 250°

These mappings are constant across all training.

A standard RNN can solve this by storing the mapping in recurrent weights.

## Plasticity Version

At the start of every lifetime:

cue_A → random angle
cue_B → random angle
cue_C → random angle

The network receives feedback after each trial.

Goal:

learn the mapping within the lifetime.

## What this tests

* fast associative learning
* memory across trials
* learning from sparse feedback

## Prediction

Fixed-weight RNN:

* poor adaptation

Plastic network:

* strong advantage

---

# T21: PairedAssociation

## Current Version

Two cue-angle pairs are presented.

cue_A ↔ angle_A
cue_B ↔ angle_B

A cue is probed later in the same trial.

## Plasticity Version

Associations persist across multiple trials.

Trial 1:
learn cue_A ↔ angle_A

Trial 2:
learn cue_B ↔ angle_B

Trial 3:
probe cue_A

## What this tests

* storage of arbitrary associations
* resistance to interference
* capacity limits

## Prediction

Fast synaptic storage may outperform persistent neural activity.

---

# T22: ReversalLearning

## Current Version

Not yet implemented.

## Plasticity Version

Lifetime begins with:

cue_A → left
cue_B → right

At a hidden reversal point:

cue_A → right
cue_B → left

The network receives only reward feedback.

## What this tests

* adaptation
* flexibility
* updating old memories

## Prediction

Strong benchmark for biologically plausible learning rules.

---

# T23: OnlineLinearRegression

## Current Version

Network sees example pairs and predicts y.

## Plasticity Version

Each lifetime contains a new function.

y = ax + b

Parameters change every lifetime.

The network must infer the function online.

## What this tests

* learning to learn
* rapid parameter estimation
* generalization from examples

---

# Open Questions

1. Which tasks truly require plasticity?
2. Which tasks can be solved with recurrent dynamics alone?
3. What neural motifs emerge before plasticity is introduced?
4. Does plasticity replace working-memory attractors or complement them?
5. How does performance scale with delay length and number of associations?
