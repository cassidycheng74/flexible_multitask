# T20: CueAssoc (Prototype)

## Motivation

The original Driscoll task suite contains no associative learning tasks. Existing tasks test working memory, decision making, context gating, and stimulus-response mappings, but they do not require the network to bind arbitrary identifiers to arbitrary responses.

CueAssoc is intended as the simplest associative task that can be implemented within the existing Driscoll framework.

## Computation

Given a cue identity, produce the angle associated with that cue.

The network must learn a mapping:

cue_i → response_angle_i

Unlike DelayGo or DelayAnti, the correct response cannot be determined directly from the stimulus itself. Instead, the stimulus serves as an identifier that selects an associated output.

## Input Format

* Fixation input active during fixation and delay epochs.
* Cue presented as a circular stimulus on modality 1.
* Cue identity represented by one of N discrete cue locations.

## Output Format

* Fixation output active before response.
* Circular response output encoding the angle associated with the cue.

## Epoch Structure

FIX → CUE → DELAY → GO

### FIX

Maintain fixation.

### CUE

Cue identity is presented.

### DELAY

Cue disappears. Association must be maintained.

### GO

Network outputs the associated response angle.

## What It Tests

* Associative retrieval
* Memory for arbitrary mappings
* Representation of discrete identities

## Relation to Future Plasticity Work

This prototype is fully supervised.

The eventual version will assign a new cue→response mapping each lifetime. The network will need to learn the mapping through experience rather than memorize it in fixed weights.

# T21: PairedAssociation (Prototype)

## Motivation

PairedAssociation extends CueAssoc by requiring within-trial associative binding.

Instead of retrieving a fixed mapping, the network must temporarily bind two cue-angle pairs and later retrieve the correct angle when probed with one of the cues.

This is closer to episodic memory and variable binding.

## Computation

Store:

cue_A ↔ angle_A

cue_B ↔ angle_B

Later, given a cue, retrieve its paired angle.

## Input Format

* Fixation signal.
* Cue identity represented as a circular stimulus on modality 1.
* Associated angle represented as a circular stimulus on modality 2.

## Output Format

* Fixation output before response.
* Circular response corresponding to the angle paired with the probe cue.

## Epoch Structure

FIX → PAIR_A → DELAY → PAIR_B → DELAY → PROBE → GO

### FIX

Maintain fixation.

### PAIR_A

Present cue_A and angle_A simultaneously.

### DELAY

Short maintenance period.

### PAIR_B

Present cue_B and angle_B simultaneously.

### DELAY

Maintain both associations.

### PROBE

Present one cue only.

### GO

Output the angle associated with the probe cue.

## What It Tests

* Variable binding
* Associative retrieval
* Multi-item working memory
* Interference between stored associations

## Relation to Existing Driscoll Tasks

This task is more demanding than MemoryPro because the network must store relationships between objects rather than a single continuous variable.

Unlike DMS/DMC tasks, there is no predefined rule that determines the correct response. The response depends entirely on temporary associations established during the trial.

## Relation to Future Plasticity Work

The current version can likely be solved using recurrent dynamics alone.

Future versions may:

* increase the number of cue-angle pairs,
* increase delay lengths,
* introduce distractors,
* allow associations to persist across trials,

making fast plasticity mechanisms more advantageous.
