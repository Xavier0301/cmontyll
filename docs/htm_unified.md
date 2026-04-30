# HTM unified — design reference

A short reference to the Hierarchical Temporal Memory networks used in this repo, the Thousand Brains theory they sit inside, and the unification proposed in Xavier's master's thesis ([thesis.pdf](../thesis.pdf), §4.1) that this repo's `layer.{h,c}` implements.

## 1. Thousand Brains, in one paragraph

The neocortex is built from ~200,000 functionally identical units called **cortical columns**. The Thousand Brains Theory (Hawkins et al., 2016–2024) proposes that each column independently learns a complete sensorimotor model of the world from its own narrow sensor patch, and that intelligence emerges from the consensus across all columns rather than from a hierarchical pipeline. A Thousand Brains *System* is an artificial system built on this principle: many semi-independent **Learning Modules** (LMs), each fed by a **Sensor Module** (SM), exchange lateral votes to converge on object identity and pose. There is no batched training over a static dataset — modules learn online, continually, by interacting with the environment.

Inside each LM, the cortical-column substructure is approximated by a small stack of HTM-style cell layers. In this repo:

| Layer | Role | Activity                                                          |
| ----- | ---- | ----------------------------------------------------------------- |
| Pooler (SM)   | Spatial encoding   | Top-k of feedforward overlap + homeostatic boosting       |
| L4 (feature)  | Sensory features   | Burst-or-predicted, distal context from self + L6         |
| L6 (location) | Object-centric location (grid cells) | Same as L4 + a movement-driven grid shift before project |
| L3 (output)   | Object identity / pose vote | Top-k of distal-spike score, gated by L4 feedforward |

The four components share a near-identical computational skeleton — so much so that they can be implemented as a single layer type with four small configuration differences. That is the unification this repo pursues.

## 2. The HTM neuron and the active-dendrite model

HTM departs from the point-neuron model used in deep learning. A cell has three input territories:

- **Proximal dendrites** receive feedforward input. Enough proximal overlap drives a somatic action potential — i.e., the cell fires (`a^t_I = 1`).
- **Distal dendrites** receive context (lateral, top-down, recurrent). Enough distal overlap on a single segment generates a local *NMDA spike* that depolarizes the soma without firing it. A depolarized cell is "predictive" (`π^t_I = 1`).
- A predictive cell that subsequently receives proximal input fires earlier than its non-predictive neighbors and **inhibits** them — this is the source of HTM's bursting behavior.

Each cell carries a small number S of distal **segments**; each segment owns C synapses with associated **permanence** values in [0, 1] (encoded as u8 here). A synapse is *connected* iff its permanence ≥ p\*. Learning is Hebbian on the permanences, not on which synapse exists — synapses are randomly initialized once and never grown or pruned in this implementation.

Activity is **sparse-distributed**: ~2% of cells active per step. Sparsity yields large representational capacity, robust noise tolerance, and crucially makes the hot path of the algorithm work over a tiny fraction of the network.

## 3. The unified Table 4.2

The thesis collapses Pooler / L4 / L6 / L3 onto five common operations. G stands for any connection set — proximal F or distal D.

| Operation               | Definition |
| ----------------------- | ---------- |
| Synaptic overlap        | μ(G, p\*) = \|{(I, p) ∈ G : p ≥ p\*, a\_I^t = 1}\| |
| Segment NMDA spike      | τ\_ij = Σ\_d 𝟙{μ\_t-1(D\_ijd, p\*) ≥ θ\_d} |
| Cell depolarization     | π\_I = 𝟙{Condition(τ\_I)} |
| Somatic action potential| a\_I = Decision(μ(F\_I, p\*), π\_I^{t-1}) |
| Hebbian learning        | p ← p ± δ for connections in update set X\_t, signed by incident activity a\_{t-1}^x_I |

The four cell-style layers differ **only** in (i) which streams are proximal vs. distal, (ii) the predicate `Condition(τ)` for depolarization, (iii) the predicate `Decision` for action potential, and (iv) which segments end up in the per-step update set.

### Per-layer instantiation

| Layer | Proximal F | Distal D | Condition(τ) | Decision (a) | Learning |
|---|---|---|---|---|---|
| **Pooler** | encoded sensory bits | — | — | μ(F) ≥ θ\_F **and** in top-k by μ(F)·boost | Hebbian on F of all active cells |
| **L4 (feature)** | active columns from pooler (gate) | self-feature + location | τ ≥ 1 | column active **and** (predicted **or** burst-all) | reinforce spiking segments of correctly-predicted cells; reinforce best segment of winner cell in bursting columns |
| **L6 (location)** | active columns from movement-shifted prev (gate) | self-location + feature | τ ≥ 1 | column active **and** (predicted **or** burst-all) | same as L4 |
| **L3 (output)** | features from L4 | self-output + external LM outputs (+ optional top-down) | top-k by τ | μ(F) ≥ θ\_F **and** τ ≥ k-th largest τ | reinforce spiking segments of cells that are active and predicted; F-permanences static (TODO) |

"Bursting" means: when a column is active but no cell in it was predicted, *all* cells in the column fire. This is HTM's mechanism for representing "I see an active column for the first time in this context — pick the best-match cell to specialize on this context next time."

The "winner cell" in a bursting column is the cell whose segments collectively saw the most active connections last step (a proxy for "closest to having predicted, but not quite enough"). That cell's best segment then receives a Hebbian reinforcement so that this exact context will be recognized next time.

### The learning update set

Per step, three kinds of segment events trigger a permanence update:

1. **Correctly predicted**: cell is `active && predicted`, segment was spiking → reinforce: increment connections from active-prev incidents, decrement those from inactive-prev incidents.
2. **Bursting-column winner**: column is `active && no cell predicted`, on the column's winner cell, on its best-match segment → reinforce by the same rule.
3. **Predicted-but-inactive (decay)**: cell is `predicted && !active`, segment was spiking → mild decay on connections from active incidents (weakens future false-positive predictions).

Items 1 + 2 cover thesis equations 4.12 and 4.13 (set Y∪Z); item 3 implements the decay rule (an extension over the base Numenta TM).

## 4. Implementation map — `layer_t`

The unified [`src/layer.h`](../src/layer.h) / [`src/layer.c`](../src/layer.c) realizes Table 4.2 as a single struct + a small handful of phase functions.

The key building blocks:

- **`input_stream_t`** — a typed pointer into someone else's activity bitarray (proximal vs. distal, with thresholds and segment quota). Top-down feedback is just another `STREAM_DISTAL`.
- **`connection_t[]`** — a single arena of (source\_index, segment\_index, permanence, stream\_id) tuples per layer. Both indices below point into this arena; permanence is owned by it.
- **by-pre CSR index** — for each (stream, source\_cell), a contiguous slice of connection ids. Used by `project()` for sparse iteration over active source cells.
- **by-seg CSR index** — for each segment, a contiguous slice of connection ids. Used by `learn()` for per-segment Hebbian updates.
- **`segment_meta_t`** — small per-segment struct that survives phase boundaries (carries the spike count from `predict` to `learn`).
- **`decision_kind_t`** — enum dispatched in `layer_decide` for the layer-specific somatic-potential rule (`DECIDE_BURST_OR_PREDICTED`, `DECIDE_TOPK_PREDICTED_FFWD`).

| Table 4.2 row | Code |
| --- | --- |
| Synaptic overlap μ(D)         | `layer_project` writes `segment_accumulators` |
| Segment NMDA spike τ          | `layer_predict` reads accumulators, writes `segment_meta.spike_count` and per-cell τ |
| Cell depolarization π         | `layer_predict` writes `predicted` bitarray (≥1 spike) — top-k version deferred to `decide` |
| Somatic action potential a    | `layer_decide` switches on `decision_kind_t`, writes `active` bitarray + sparse list |
| Hebbian update                | `layer_learn` builds the update set, walks by-seg index, applies ±δ |

L6's grid-cell shift is `layer_shift_active_prev_grid()` — a pre-`project` hook that permutes the layer's own previous active cells along a movement vector before they're projected. The unified API stays unaware.

## 5. Storage trade-off

Two competing layouts exist: per-segment ("for each segment, the list of connections coming in") and per-cell ("for each presynaptic cell, the list of connections going out"). The first makes learning natural; the second makes prediction sparse. With ~2% sparsity, the second wins by 50× on the project hot path. But Hebbian learning on a single segment requires touching all of its connections — natural in the first layout, an awkward back-walk in the second.

The repo resolves this with a **single connection arena, two CSR indices**. Permanence lives once in `connection_t`; both indices contain only u32 connection-id pointers into the arena and are read-only after init. Cost: roughly 1.6× memory vs. either single-layout choice. Benefit: cache-linear sparse iteration in *both* `project` (active source cells × by-pre) and `learn` (update-set segments × by-seg).

For a representative L4-style layer (cols=1024, cells=8, segments=12, connections-per-segment=30), this works out to ≈ 60 MiB per layer, dominated by the 12-byte `connection_t` and the two 4-byte index arrays.

## 6. Extending: top-down, multi-LM, and the next steps

Top-down feedback (e.g., L3 → L4) is a one-line addition: register an extra `STREAM_DISTAL` on the destination layer with `activity = source_layer.active`, allocate a small `segments_assigned` quota, and `init` will draw connections that integrate naturally into the unified prediction step.

Cross-LM connections (used by the L3 output layer's external context) follow the same pattern. The source for those is the activity bitarray published by another LM's L3, accessed via the existing `lmat` indirection.

Open hooks not implemented in v1:

- L3 feedforward learning (legacy TODO at [output_layer.c:256](../src/output_layer.c)). The proximal arena has the symmetric by-seg index already, so flipping `enable_ffwd_learning` is a one-line change in `learn`.
- Topology-aware proximal connections (currently a uniform random subset).
- Multi-threading. `project` parallelizes per active source cell; `learn` parallelizes per update-set segment.
- Bridge to the sibling `montx-htm/` system. The unified `layer_t` produces and consumes plain `u32*` packed-32 bitarrays — the bridge is a thin SDR adapter, slated for the next session.

## References

- Thesis [§2.1, §4.1.1–4.1.5](../thesis.pdf) (Servot, 2025) — primary source for the unification.
- Hawkins, Ahmad, Cui (2017) — Why neurons have thousands of synapses (active dendrite model).
- Hawkins, Ahmad (2016) — Why does the neocortex have layers and columns? (HTM cell, bursting, TM).
- Lewis, Purdy, Ahmad, Hawkins (2019) — Locations in the neocortex (grid cells in L6).
- Cui, Ahmad, Hawkins (2017) — The HTM Spatial Pooler.
- Hawkins (2021) — *A Thousand Brains*.
- Leadholm et al. (2024) — Monty: a thousand-brains-style system.
