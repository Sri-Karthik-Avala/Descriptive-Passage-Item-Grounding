# Descriptive Passage Item Grounding

| | |
| --- | --- |
| Final rank | not ranked |
| Domain | NLP |
| Difficulty | Medium |
| Scoring | ↑ Higher is better |
| Compute | A10G |
| Challenge status | Accepted / closed |
| Solutions submitted | 4 |
| Last submission | 2026-08-04 |

## Problem statement

### Overview

Each task presents one **descriptive passage** — a short piece of text written about a single real-world item — together with a **fixed slate of 20 candidate items**. Exactly one candidate is the item the passage was written about; the other 19 are decoys. A system must output a **calibrated probability distribution** over the slate: its belief that each candidate is the item the passage describes.

The passage never names the item and never repeats its identifiers — it only describes the *experience* of the item in free text. Each candidate is represented solely by its **content**: a tag profile, a locale, and a short content summary distilled from independent text about it. The difficulty is that the 19 decoys are chosen to be the same-category candidates whose content summary is the **closest surface-lexical match to the passage itself** — the strongest word-overlap distractors — so that shallow term overlap between the passage and a candidate's summary points at the decoys as much as at the true item, and the surface tags and text length shared across the slate do not identify it.

Tasks are stratified into five **difficulty tiers** by how tightly the decoys crowd the true item on this passage-to-content match — from tiers where the true item is the clear surface match, to tiers where the decoys match the passage as well as or better than the true item does — and the five tiers are weighted equally, so a system must ground passages against easy and adversarial slates alike. Because the required output is a full belief distribution rather than a single pick, scoring rewards **calibrated** confidence: mass on the true item earns credit, an even spread earns the chance baseline, and confident mass on a wrong item is penalized.

### Dataset

### Public files

- **`items.csv`** — one row per candidate item. Columns: `item_id`, `tags`, `locale`, `region`, `tier`, `traits`, `content_summary`.
- **`train.csv`** — labeled tasks. Columns: `task_id`, `passage`, `difficulty`, `slate`, `chosen_item_id`.
- **`test.csv`** — unlabeled tasks. Columns: `task_id`, `passage`, `difficulty`, `slate`.
- **`sample_submission.csv`** — a valid submission in the required format (uniform distributions).

### Private file (organizer only)

- **`answers.csv`** — columns: `task_id`, `difficulty`, `target_position` (1-based index of the described item in the slate). Used only for scoring; never distributed.

### Column descriptions

Every column in the public files is described below.

- **item_id** (string) — unique item identifier (e.g. `item_3a9f1c…`).
- **tags** (string) — comma-separated content tags for the item (e.g. `Coffee, Breakfast, Cafe`).
- **locale** (string) — the locale the item belongs to.
- **region** (string) — coarse region code.
- **tier** (integer) — item price/level tier 1–4, or `-1` if unknown.
- **traits** (string) — semicolon-separated `key=value` item traits (e.g. `OutdoorSeating=True;GoodForKids=False`).
- **content_summary** (string) — a short concatenation of snippets of independent text about the item (`|`-separated), describing what the item is like. This is the sole content signal for grounding.
- **task_id** (string) — unique task identifier; one row per task and one submission row per `task_id`.
- **passage** (string) — the descriptive text to ground; written about the true item, with its name and identifiers removed.
- **difficulty** (string) — the difficulty tier of the task, one of `t1`, `t2`, `t3`, `t4`, `t5` (increasing crowding of the decoys around the true item in passage-to-content surface match; `t5` is hardest).
- **slate** (string) — the candidate slate: exactly **20** `item_id`s separated by `;`, in a fixed order. Your probabilities are aligned to this order.
- **chosen_item_id** (string, train only) — the `item_id` in the slate the passage describes. Present in `train.csv` only.

Each `item_id` in a task's `slate` has its content row in `items.csv`. The `difficulty` field indicates how strongly the decoys compete with the true item on passage-to-content surface match.

### Data example

A truncated `train.csv` row:

```
task_id,passage,difficulty,slate,chosen_item_id
task_tr000001,"cozy spot for a quick espresso before work, friendly staff …",t3,item_a;item_b;item_c;…;item_t,item_c
```

### Submission format

Submit a CSV named `submission.csv` with **exactly one row per test `task_id`** and **exactly these 21 columns**:

```
task_id,p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13,p14,p15,p16,p17,p18,p19,p20
```

- `pk` is the probability that the **k-th item in that task's `slate`** (fixed order) is the item the passage describes. Values are non-negative floats.
- The 20 probabilities in each row must sum to `1.0` (tolerance `±0.02`; values are renormalized before scoring).
- Every test `task_id` must appear exactly once. Submissions with missing ids, unknown ids, duplicate ids, extra columns, missing columns, negative values, or non-finite values are rejected.

Sample submission (uniform belief on every row):

```
task_id,p1,p2,…,p20
task_te000000,0.05,0.05,…,0.05
```

### Evaluation

**Metric: Tier-Balanced Calibrated Grounding Score (higher is better).**

Each task's predicted distribution is scored by a blend of a **ranking** term (does the true item rank near the top of the slate?) and a **calibration** term (a strictly-proper log score on the probability mass placed on the true item). Per-task scores are averaged **within** each of the five difficulty tiers, and the metric is the **mean of the five tier averages**, so every tier — including the hardest — contributes one fifth regardless of how many tasks it has.

For a task with predicted distribution `p` over the 20 slate positions and true position `t`:

```
p = renormalize(p)                                  # sum to 1
g = (# of positions with p > p[t])                  # strictly greater
k = (# of positions with p == p[t])                 # size of the tied block (includes t)

# expected reciprocal rank over the tied ranks g+1 .. g+k (what a random
# tie-break scores in expectation), so information-free jitter buys nothing:
rr   = mean(1.0 / r for r in range(g + 1, g + k + 1))          # in [1/20, 1]
cal  = 1.0 + math.log(max(p[t], 1e-9)) / math.log(20.0)        # normalized log score
cal  = min(1.0, cal)                                # uniform -> 0, certain-correct -> 1, confident-wrong -> negative

base = 0.5 * rr + 0.5 * cal                          # per-task score (can be negative)
```

Aggregate:

```
TIERS = ["t1", "t2", "t3", "t4", "t5"]
tier_mean_r = mean(base_i for tasks i in tier r)     # for each tier r
score = mean(tier_mean_r for r in TIERS)             # equal-weight macro-average
score = max(0.02, min(1.0, score))                   # floor 0.02, ceiling 1.0
```

Component summary:

- **rr** — reciprocal rank of the true item in the predicted ordering; ties are credited at the **expected reciprocal rank over the tied positions** (the mean of `1/r` across the ranks the tied block occupies), so an even spread scores the chance rank and adding information-free noise to break ties gives no advantage. Rewards ranking the true item near the top of the slate.
- **cal** — a strictly-proper normalized log score on the probability assigned to the true item; `1/20` (uniform) maps to `0`, certainty on the truth maps to `1`, and placing little mass on the true item while being confident elsewhere drives it **negative**. Capped above at `1` (no lower cap), so overconfident-wrong predictions are penalized; the final macro-averaged score is floored at `0.02`.
- **base** — the equal blend `0.5·rr + 0.5·cal` per task.
- **score** — the equal-weight mean of the five per-tier averages, clamped to a floor of `0.02` and ceiling `1.0`.

**Baseline (uniform / random) expected score:** ≈ `0.09`. A constant-uniform submission and a random-noise ranking both score the chance baseline of ≈ `0.09` (the calibration term is `0` for a uniform belief, and the expected reciprocal rank of the true item over a 20-item slate is `(1/20)·Σ_{r=1..20} 1/r ≈ 0.18`, so `base ≈ 0.5·0.18 ≈ 0.09`). **Higher is better.**

### What Not To Use (Prohibited Methods)

This challenge measures semantic grounding of a descriptive passage to the item it describes. The following are prohibited:

- **No external text-to-item lookup.** Do not retrieve, scrape, or reconstruct the original source of any passage, and do not match a passage, `item_id`, item text, or locale back to an external corpus to recover which item a passage describes.
- **No id-based hardcoding.** Do not build any mapping from `task_id` or `item_id` directly to a target position or probability. Predictions must come from the provided passage and item content only.
- **No train/test leakage.** Do not use any test-set target signal, and do not tune on the private target distribution.
- **No manual labeling of the test set.** The submission must be model-generated; do not hand-assign probabilities by inspecting individual test tasks.
- **No name/identifier recovery.** Passages have their item name and identifiers stripped; do not attempt to re-derive them from external sources to shortcut the grounding.
- Generic prohibitions also apply: no external answer keys, no id-to-target lookups, no memorized external mappings.
