# What the paper would be

A concrete sketch, written against the results you actually have plus the ones
still outstanding. Read the honest-assessment section at the end first if you
only read one part.

---

## The framing that makes this a paper

You do not have "we invented a mechanism." Channel-wise kWTA is `local=True` in
`nupic.torch`. Trying to sell it as new invites one reviewer to find that flag and
end the discussion.

What you have is better-defended and rarer: **the field has been measuring this
wrong, and when you measure it correctly the conclusion changes sign.** The
mechanism becomes the vehicle, not the contribution.

That reframe costs nothing in substance and buys you a claim nobody can take away,
because it rests on your own controlled measurements rather than on priority.

---

## Title candidates

1. **Sparsity is a prior, not an architecture: quantifying what local
   winner-take-all buys in the low-label regime**
2. Fixed epochs, false conclusions: optimization budget confounds low-data
   architecture comparisons
3. What does a structural prior buy? Measuring sparse activation as a data
   multiplier

(1) leads with the positive result and keeps the methodological finding as
support. (2) leads with the methodology — stronger for a workshop on evaluation or
reproducibility, weaker for a main track. I would write (1).

---

## Abstract (draft — numbers in brackets are still outstanding)

> Sparse activation is a recurring brain-inspired proposal for improving
> generalization from limited data, but the claim is rarely tested under a
> protocol that could support it. We show that the standard practice of holding
> epochs fixed while shrinking the training set confounds architecture with
> optimization budget: a k-winner-take-all CNN appears to lose to a dense baseline
> by 5.4 points at 600 MNIST labels, and 71% of that gap disappears once both
> models are trained to convergence and scored at their best-validation
> checkpoint. Under the corrected protocol we find that the *axis* of competition,
> not the amount of sparsity, determines whether sparsity helps: at matched total
> sparsity, competition among channels sharing a receptive field beats the dense
> baseline by 1.4 points at 600 labels (5/5 seeds, p < 0.001; a 1.8x label
> multiplier), while a global top-k over the flattened feature tensor loses by 1.6.
> The benefit is confined to the low-label regime and decays predictably. Sparse
> models produce measurably better pattern separation than dense ones (2.2x at 600
> labels), but that advantage falls monotonically to nothing by 30k labels, by
> which point dense networks have discovered comparably sparse and separated codes
> unaided. We conclude that architectural sparsity is best understood not as a
> better architecture but as a prior that substitutes for data, whose value is
> quantifiable as a data multiplier that decays with label count and crosses one at
> a measurable budget [and, on CIFAR-10, at budget X]. We additionally report two
> negative results: duty-cycle boosting achieves both of its mechanical objectives
> — fewer dead units, higher separation — while costing 8 points of accuracy
> through a train/eval inconsistency; and the dead units it targets are localized
> almost entirely to the classifier layer and are not what limits the mechanism.

---

## What it actually reveals

This is the part worth caring about. Four things, in descending order of how much
they would change someone else's behaviour.

### 1. A structural prior's value is measurable, and it expires

The field argues about whether brain-inspired mechanism X "helps." The right
question is *at what data budget does it stop paying*. Expressing the effect as a
**data multiplier with a crossover point** turns an unfalsifiable style debate into
a curve. Your numbers already trace one: 1.8x at 600 labels, ~1.0x by 3,000,
below 1.0 by 30,000.

This generalizes past sparsity. Any hard-coded structural prior — weight sharing,
equivariance, sparsity, modularity — should be reportable this way, and almost
none of them are.

### 2. The baseline finds the same solution on its own

Dense networks self-sparsify with data (77 -> 46 active units of 128) and their
code separation rises to overtake the sparse model's (0.057 -> 0.260 vs 0.230).
So the prior is not teaching the network something it could not learn. **It is
saving it the data required to learn it.**

That is a far more precise claim than "brain-inspired architectures generalize
better," and it makes a falsifiable prediction: any prior worth having should show
this signature — a decaying advantage and a baseline converging toward it. A prior
whose advantage does *not* decay is either not a prior or the baseline is
mis-specified.

### 3. Low-data architecture comparisons in this literature may be systematically confounded

Fixed-epoch protocols are common. They give a mechanism that updates fewer weights
per step — which every sparse or gated method does — proportionally less
optimization exactly where data is scarcest. In your case that manufactured a
5.4-point deficit out of a 1.6-point one and inverted the sign of the conclusion
relative to the channel-wise arm.

This is the finding most likely to change what other people do, and it is the one
you can defend most completely, because it is a within-your-own-lab controlled
comparison rather than a claim about anyone's results.

### 4. A mechanism can work and still not pay

Sparse codes measurably separated patterns exactly as theorized, and accuracy did
not move. Boosting hit both of its stated mechanical targets and cost 8 points.
Accuracy alone reads both as "does nothing." With a mechanism-level metric they
read as two different, actionable situations.

The methodological recommendation: **every mechanism ships with a measurement of
the thing it claims to do, logged beside accuracy.** Cheap, and it is the
difference between a null result and a finding.

---

## Structure

| Section | Content | Status |
|---|---|---|
| 1. Introduction | Sparsity-for-data-efficiency claim; why it is under-tested | — |
| 2. Related work | LWTA / Compete to Compute; WTA autoencoders; Numenta kWTA + boosting; hippocampal pattern separation. **Cite `local=True` explicitly and position against it** | needs real lit review |
| 3. Protocol | Convergence matching, frozen val split, stratified sampling, subset normalization, paired seeds, pre-registration | ✅ built |
| 4. The confound | v1 vs v2; r = 0.92 with log steps; 71–85% recovery | ✅ have |
| 5. Axis at matched sparsity | Stage 1 factorial; +1.43 vs −1.58 | ✅ have |
| 6. What sparsity buys | Separation curves, data multiplier, dense self-sparsification | ✅ have |
| 7. Where and how much | Placement × k; optimal k vs budget | ⏳ stage1b + placement |
| 8. Scaling | CIFAR-10, same protocol | ❌ **gate** |
| 9. Negative results | Boosting; dead-unit localization | ✅ have |
| 10. Limitations | Small CNN, two datasets, not competitive with augmentation/pretraining | — |

## Figures — you already have most of them

1. **The confound.** v1 vs v2 paired delta by budget. → `protocol_v2_result.html` panel A
2. **Axis at matched sparsity.** Arm comparison with seed dots. → `stage1_result.html` panel A
3. **Separation vs budget**, sparse and dense, with the crossover. → `protocol_v2_result.html` panel B
4. **The money figure — data multiplier vs label budget**, ideally MNIST and CIFAR-10 on one axis, with the 1.0 crossing marked. ❌ *not yet built; this is the figure the paper is about*
5. Optimal k vs budget, and placement ablation. ⏳ stage1b
6. Dead units by layer + boosting's train/eval agreement. → `stage1_result.html` panel B + `diagnose_boost`

Figure 4 is the one a reader will remember. Everything else supports it.

---

## Experiments still missing

**Gating:**
- **CIFAR-10 under identical protocol.** Without it this is an MNIST paper and MNIST
  flatters every regularizer. Needs a GPU.
- **Benchmark against `nupic.torch` `KWinners2d(local=True)` directly**, not only your
  own implementation. Reviewers will ask; matching the reference implementation is
  also a correctness check on your module.

**Strongly expected:**
- **A second architecture.** One 4-layer CNN cannot support "the axis matters."
  A wider/deeper variant is enough to show it is not architecture-specific.
- **Placement ablation.** Currently you sum a harmful and a helpful placement.
- **Compute-matched control.** Show the dense baseline given the sparse model's
  wall-clock does not close the gap — otherwise "you just trained it longer."

**Nice to have:**
- Optimal k as a function of budget, fitted. If it is a clean relationship it is
  the most quotable thing in the paper.
- One non-vision task, to argue it is not a property of images.

---

## The objection that decides the paper's fate

> "1.8x fewer labels — but data augmentation gives more than that, and
> self-supervised pretraining gives 10–100x. Why would anyone use this?"

This is the question that sinks the paper if you have not answered it before a
reviewer asks. Two defensible answers, one indefensible one.

**Defensible A — this is a measurement paper about architectural priors in
isolation.** The claim is not "best label-efficiency method." It is "here is what a
structural prior is worth, measured cleanly, and here is the protocol error that
made previous estimates unreliable." Then you must *say* that in the intro, and not
compare against augmentation as if you were competing.

**Defensible B — show composition.** Run channel-wise kWTA *with* standard
augmentation and show the multipliers partly compose. A prior that still buys
something on top of augmentation is much more interesting than one that is merely
an inferior substitute. This is one extra arm and it materially strengthens the
paper; I would run it.

**Indefensible** — quietly not mentioning augmentation. Everyone in the room knows.

---

## Honest assessment

**Realistic venue.** A workshop — data-centric ML, science-of-deep-learning, or a
negative-results / evaluation venue. Not main-track NeurIPS/ICLR: the mechanism is
known, the datasets are small, and the effect is 1–2 points. That is not a
criticism. A well-executed workshop paper that changes how people run an experiment
is worth more than a main-track paper nobody reproduces.

**What could still kill it.** The effect not surviving CIFAR-10 is the big one, and
it is more likely than not — MNIST is nearly linearly separable and flatters
sparsity. If that happens you still have a paper, but it becomes paper (2): *the
protocol confound is real, the mechanism effect is MNIST-specific, here is the
evidence for both.* That is a genuine contribution and it is honest. Decide now
that you will write it either way, so the CIFAR-10 result does not become something
you are tempted to negotiate with.

**What is genuinely yours.** Not the mechanism. The protocol, the confound
measurement, the mechanism-level metrics, the data-multiplier framing, and the
discipline of pre-registering predictions and then reporting the ones that got
falsified. That last one is rarer in this literature than any architecture.

**Timeline, roughly.** stage1b + placement: a week. GPU setup and CIFAR-10: two to
three weeks. Second architecture and augmentation arm: a week. Lit review: do it
properly, a week, and do it *before* the CIFAR runs so you are not building on a
misread. Writing: two weeks. So roughly two months of evenings, assuming CIFAR-10
cooperates.
