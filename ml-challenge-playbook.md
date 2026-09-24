---
name: ml-challenge-playbook
description: General playbook for solving a "fine-tune a pretrained model, submit predictions" ML challenge (Shipd-style or similar) - workflow, CV/leakage diagnosis, metric-aware calibration, compute budgeting, and common error fixes. Domain-agnostic; no numbers or file paths from any specific past challenge.
---

# Playbook: solving a constrained ML prediction challenge

Applies to challenges of the shape: given train/test data + a fixed grader command
(`python3 solution.py <public_dir> <out>`), predict a target under a compute/time budget, scored
by some correlation- or agreement-based metric. Covers CV design, metric-aware training,
compute planning, and recurring bugs. Pairs with [[possibility-discipline-rules]] (open every
possibility, close only on a measured result) - this file is about *how* to run the experiments
that check, honesty, this is the *doing* half.

## 0. Read before touching code

1. Problem statement: what is the target, what is forbidden ("no source lookup", "no external
   data", "must fine-tune", "CPU only", etc.), what is the exact scoring formula.
2. Submission format spec: exact column names/order, value ranges, JSON-array-in-CSV quoting,
   what causes a **global** 0.0 (malformed row, wrong id set, extra/missing columns) vs a
   per-row penalty. Re-read this after every change to the output-writing code.
3. Evaluation criteria screenshot/page (if separate from the problem statement) for the runtime
   command, resource limits (cores, RAM, wall-clock), and any "test before submitting" hints.
4. Compute envelope: CPU-only vs GPU, core count, RAM, wall-clock budget. This determines the
   whole design (frozen-feature + light head vs full fine-tune, ensemble size, etc.) - measure
   it before choosing an architecture, don't guess.

## 1. First diagnose the data, don't just start modeling

- Load train/test, print shapes, dtypes, id structure, target range/mean/std/quantiles per
  output dimension, and correlations between output dimensions if there are several.
- **Check for near-duplicate / leaked rows before trusting any CV split.** Perturbed or
  augmented copies of the same underlying example are common (surveys, crops, elicited-speech
  variants, resampled record sets). Method: embed every row cheaply (a small pretrained
  encoder's pooled features, or simple hand features), compute pairwise cosine similarity within
  train, and look at the label difference between each row and its nearest neighbour. If
  near-neighbours have near-identical labels, a naive random K-fold will leak and every model
  will look far better locally than it will score for real.
- **Measure the train-vs-test distance gap**, not just the train-internal one: nearest-neighbour
  cosine distance from each test row to its closest train row, versus each train row's distance
  to its closest *non-duplicate* train neighbour. If held-out folds need to look like test, build
  them to match this profile (see next section) rather than trusting a generic K-fold.
- Only ever use test **inputs** for this kind of structural diagnosis (embeddings, shapes,
  vocabulary/char coverage) - never test labels or pooled test statistics that would let one
  test row influence another's prediction (see [[possibility-discipline-rules]] check 8).

## 2. Build a CV protocol that actually predicts the leaderboard

- If duplicates/near-duplicates exist, group rows into clusters (agglomerative clustering on the
  same embedding, average linkage, cosine distance) and use **GroupKFold on the clusters**, not
  on raw row IDs, so no fold contains a train row and its twin split across train/val.
- Sweep the clustering distance threshold and pick the one whose held-out-to-train nearest-
  neighbour distance distribution matches the measured test-to-train distribution. Matching that
  *profile* does not automatically guarantee it matches the *leak level* end-to-end - after
  building the protocol, sanity-check by comparing a fixed model's OOF score across two
  candidate protocols; a big gap means one of them still leaks.
- Track results **per fold**, not just averaged. If clustering makes one fold consistently much
  harder (e.g. it happens to contain the largest, most distinctive cluster), that fold is your
  best available proxy for a distribution-shifted real test set - use it to compare architectures
  and regularization, not just the mean.
- Treat every non-final-graded number as an **unvalidated proxy**, always. Local CV, held-out
  splits, "should transfer" reasoning - none of it is the real result until the grader says so.

## 3. Metric-aware modeling

- Read the scoring formula precisely and ask: does it reward correlation only, or also
  reward/penalize a magnitude/variance mismatch? (Examples: CCC, weighted kappa, and most
  "concordance" metrics penalize a well-correlated but badly-scaled prediction; plain Pearson/AUC
  do not care about scale.) If the metric penalizes scale mismatch, ridge/tree regressors with
  strong shrinkage will systematically under-predict variance and lose real points that a simple
  post-hoc rescale would recover for free.
- **Fit scale/shift calibration on out-of-fold predictions only**, never on the training targets
  directly through the model itself and never on test statistics. Multiply the centered OOF
  prediction by (label std / OOF-prediction std) per output dimension, then add the label mean.
  Apply the same fitted scale/shift to the test predictions.
- When one fold looks like the best available proxy for real-world shift (see above), consider
  a **hedged scale**: the geometric mean of the scale fitted on all OOF rows and the scale fitted
  on that hard fold's rows only. This trades a small amount of average-case CCC for robustness on
  the shifted portion, which is presumably closer to what "generalization" means to the grader.
  Verify the trade-off explicitly (report both the overall and the hard-fold CCC before and
  after) rather than assuming it helps.
- If a custom loss matching the metric exists (e.g. a differentiable CCC loss), train the model
  head directly on it rather than plain MSE - it noticeably changes what the model spends
  capacity on for these variance-sensitive metrics.

## 4. Modeling strategy under a tight compute budget

- **Start from frozen pretrained representations before fine-tuning end-to-end.** Extract
  all-layer hidden states once, cache them, and try (a) mean/std-pooled features -> linear/ridge
  regression, (b) a small sequence head (attention pooling, a couple of transformer/conv/GRU
  layers) on the raw or layer-averaged sequence. This is almost always cheaper and often
  competitive with full fine-tuning, especially with a small training set.
- **Layer selection matters more than architecture choice** for frozen features: middle layers
  of a self-supervised encoder are usually more task-transferable than the very first or very
  last layers. Sweep single layers, then try averaging/concatenating small contiguous groups of
  layers rather than either one layer or all layers uniformly.
- **Seed-average the head**, not the backbone: light heads on frozen features have high seed
  variance (can be several points of the metric); 3-5 seeds per fold is usually enough to make
  the ranking of design choices trustworthy. Do this before drawing conclusions from a single run.
- Simple regularization sweeps (dropout, weight decay, mixup) should be evaluated **on the
  hardest/shifted fold specifically**, not just the overall average - a setting that trades a
  little average performance for much better behavior on the shifted fold is often the right
  choice for a real, disjoint test set, per check 14(b) of [[possibility-discipline-rules]]
  (prefer the broad, robust plateau over a swept argmax).
- End-to-end fine-tuning of the full backbone is usually only worth it if (a) the frozen-feature
  approach is clearly compute-constrained relative to the budget, and (b) a controlled comparison
  actually shows it beating the frozen approach on the same CV protocol - don't assume "more
  trainable parameters = better" under a small, twin-heavy training set.
- **Never retreat to a lower-fidelity method family just because the strongest member of a
  family failed once or is expensive.** Try a cheaper variant within the same family (fewer
  layers unfrozen, smaller head, time-pooling to shrink sequence length) before abandoning the
  family outright (check 3/5/16 of [[possibility-discipline-rules]]).

## 5. Ensembling

- Build a small library of independently-trained members (different backbones, different heads,
  different layer subsets) with their out-of-fold and test predictions saved separately.
- A **uniform average of several honestly-diverse members** is a strong, low-variance baseline.
  Greedy forward selection on the OOF metric can look better locally but easily overfits the CV
  split, especially with few folds - prefer the uniform blend, or a small greedy blend validated
  on a metric that wasn't used to pick the members, unless there's a strong, robust reason to
  down-weight/drop a member (e.g. it is measurably and consistently the weakest across folds).
- Fewer, individually stronger members often beats many marginal ones. Periodically re-check
  whether pruning the weakest member(s) improves rather than hurts the blend.
- Apply the metric-aware calibration (section 3) to the **blended** OOF prediction, once, at the
  end - not per-member before blending.

## 6. Compute and time budgeting

- Measure actual throughput on the target hardware/thread-count early (a tiny timing script:
  N forward/backward passes at the real batch size), rather than assuming a paper's numbers or
  a different machine's speed transfer. Recompute the total wall-clock estimate (folds x seeds x
  epochs x per-step time) and compare it against the grader's stated budget with real margin.
- If a full end-to-end script is close to the wall-clock budget, prefer: fewer/smaller heads,
  temporal or spatial pooling to shrink sequence/feature length, precomputing anything that is
  reused across seeds/folds, and fewer redundant model checkpoints - in that order - rather than
  cutting corners on validation (fewer folds, no calibration, etc.).
- Long local jobs should run **detached** from the interactive session (a background/queued
  process that survives the assistant session ending), with progress logged to a file that gets
  polled rather than blocking on it. Keep the machine from sleeping/throttling during long runs.
- When sharing a GPU or CPU pool with other concurrent work, expect variable per-step timing;
  don't over-index on a single wall-clock measurement taken under contention.

## 7. Validate the submission mechanically, every time

Before trusting any generated submission file:
- Row count matches the test set exactly; id set matches exactly (no missing/extra/duplicate
  ids); columns match the required name and order exactly.
- Every value is finite, within any stated numeric range, and the exact type/shape the spec
  requires (e.g. a fixed-length JSON array with the right element count and no non-finite
  values). A single malformed row can zero the *entire* submission under many grading schemes -
  treat this validation as mandatory, not optional, and run it as a separate automated check
  rather than eyeballing a few rows.
- Diff a new candidate submission against the previous one (row-wise correlation per output) to
  catch large unintended regressions before submitting.
- Actually run the exact grader invocation (`python3 solution.py <public_dir> <out>`) end-to-end
  locally at least once before shipping, including a full untouched copy of the public data, not
  just the training loop in isolation - integration bugs (path handling, argv parsing, package
  imports available only in dev) are common and easy to miss otherwise.

## 8. Diagnosing "local proxy scored much higher than the real grade"

If the local CV/holdout number and the real graded score disagree by a lot:
1. First suspect **leakage in the CV protocol** (near-duplicates split across train/val,
   pooling across what should be independent rows) - this is the single most common cause of a
   large, one-directional gap.
2. Check whether the **hardest fold** (the one built to resemble the real distribution shift) is
   much closer to the real score than the average fold is. If so, the average was always
   optimistic and the hard fold should become the primary internal metric going forward.
3. Re-verify the calibration scale wasn't fitted in a way that leaks test information, and that
   it doesn't overshoot when applied to genuinely different real-world data (a fold-hedged scale
   estimated from section 3 mitigates this).
4. Confirm the shipped model/version is actually the one that was validated (stale predictions,
   wrong checkpoint, an untracked code edit between validation and shipping are easy to miss
   under time pressure).
5. Before assuming there is a bug at all, check whether *any* published leaderboard result is
   close to the local estimate - if nobody is near it, the local scale may simply not be the real
   scale, and there may be nothing to fix (this is check 17 of
   [[possibility-discipline-rules]]).

## 9. Iteration discipline while racing a leaderboard

- Change **one variable per submission** once real feedback starts coming in, especially when
  time-constrained - otherwise a real regression and a real improvement can cancel out and teach
  nothing (check 18 of [[possibility-discipline-rules]]).
- Keep shipping improving versions rather than holding out for a "final" one: a validated,
  submitted improvement beats an unsubmitted, purely-local one.
- Record what was tried and closed (with the actual measured result) so a later session - or a
  fresh chat with no memory of this run - doesn't repeat a rejected experiment. Generalizable
  lessons (this file) belong separately from a specific challenge's numeric results.

## 10. Common environment/library gotchas worth checking early

- Framework API drift: check whether a library's internal layer/module call returns a bare
  tensor or a tuple in the installed version before indexing into it (`layer(x)[0]` silently
  mis-indexes if `layer(x)` is already a tensor). Print `type(...)` once when wiring up any
  "reach into a pretrained model's internals" code path.
- On Windows specifically: no case-sensitive filesystem assumptions; PowerShell has no `&&`
  chaining (use `;` or `if ($?) {}`); avoid naming any local helper module the same as a stdlib
  module name (shadowing causes hard-to-diagnose import failures deep in a dependency); detached/
  background processes need `Start-Process` (or equivalent) to survive the parent session ending;
  disable sleep/standby for long unattended runs.
- Dataloader/worker code must guard module-level side effects (building caches, downloading
  files) behind `if __name__ == "__main__":` or an explicit worker-init check, or it silently
  re-runs per worker process.
- Re-verify package availability and versions in the *actual* execution environment, not assumed
  from familiarity with the library - APIs (argument names, return types, config defaults) drift
  between versions in ways that break specific call patterns.
