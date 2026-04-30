# Integration guide — porting `layer_t` to `montx-htm`

You are continuing work in a sibling project (`montx-htm/`) where another agent has built a thousand-brains-style system around a different stack, and now wants to slot in this repo's unified HTM `layer_t` so a learning module can be composed of HTM parts. **You don't need to read the rest of this repo to do that.** This document is the contract.

If you want the *why* behind the design, read [`htm_unified.md`](htm_unified.md) — three pages distilling thesis §4.1 and the storage trade-off. The integration itself does not require it.

## 1. What `layer_t` is

A single C type that realizes the four cell-style HTM layers (Pooler aside) from one struct + ~10 phase functions, configured by one `decision_kind_t` enum and a list of `input_stream_t` descriptors.

Inputs: any number of packed-32 bitarrays (proximal or distal), each tagged with source dimensions and per-stream thresholds. The layer reads — never writes — them.

Outputs: a packed-32 `active` bitarray, plus a sparse `active_cells[]` list. Both are layer-owned and stable until the next step.

Lifecycle: two-phase init (state, then connections) so self-recurrent or mutually-recurrent streams can point at activity buffers that already exist; one `layer_step(...)` per simulation step (or four manual phase calls for layers with custom pre-projection hooks like L6's grid shift).

## 2. Minimal file set to port

**Mandatory** (copy as-is into `montx-htm/src/`):

| File | Lines | Role |
|---|---|---|
| `src/layer.h`           | 354 | public API |
| `src/layer.c`           | 999 | the unified layer |
| `src/lm_parameters.h`   | 119 | `htm_params_t`, `extended_htm_params_t`, `make_l4/l6/l3_params` |
| `src/lm_parameters.c`   | 151 | constructor bodies |
| `src/types.h`           |  45 | `u8`/`u32`/`vec2d`/`uvec2d` |
| `src/tensor.h`          |  89 | `tnsr_u8` (used by `segment_accumulators`) |
| `src/tensor.c`          |   9 | tensor init |
| `src/bitarray.h`        |  42 | packed-32 macros (mostly historical; `layer.h` ships its own helpers) |
| `src/lmat.h`            |  59 | `lmat_u32` (placeholder; only the type is referenced) |
| `src/lmat.c`            |  19 | matching |
| `src/distributions.h`   |  43 | `unif_rand_range_u32` |
| `src/distributions.c`   | 154 | RNG implementation |
| `src/algorithms.h`      |  24 | only `safe_add_u8` / `safe_sub_u8` are used |
| `src/algorithms.c`      |  47 | matching |
| `src/location.h`        |  18 | `vec2d`, `uvec2d` (used by L6 grid shift) |

That is the complete dependency closure of `layer_t`. ~2200 lines total. Run `cc -Wall -Wextra -O2 -c src/layer.c` after copying — it should produce zero warnings.

**Recommended** (a worked composition example — copy if you want a learning-module wrapper analogous to this repo's `learning_module.{h,c}`):

| File | Lines |
|---|---|
| `src/learning_module.h` | 95 |
| `src/learning_module.c` | 275 |

These show how three `layer_t` instances are wired into the L4 / L6 / L3 stack of a single column, including the L6 grid-shift phase, the L4 column gate from a pooler-style upstream, and the L3 top-k decision.

**Tests** (also self-contained):

| File | Lines |
|---|---|
| `tests/test_layer.c` | ~370 |

Per-phase sanity tests (project, predict, decide BURST, decide TOPK, learn REINFORCE, learn DECAY, learn saturation, grid shift). Eight tests / 23 assertions; run with `make test`. The tests double as canonical fixture-construction examples — read them when you want to learn the API by example.

## 3. The contract (one screen)

```c
/* ---- shape ---- */
typedef struct layer_params_t_ {
    u16 cols;                      /* per-column layout if cells_per_col > 1; flat if == 1 */
    u8  cells_per_col;             /* 1..32 */
    u8  segments_per_cell;
    u16 conns_per_segment;
    u16 ffwd_conns_per_cell;       /* 0 if no proximal stream */
    decision_kind_t decision;      /* DECIDE_BURST_OR_PREDICTED  or  DECIDE_TOPK_PREDICTED_FFWD */
    u16 top_k;
    htm_params_t htm;
    u8  enable_distal_learning, enable_decay, enable_ffwd_learning;
} layer_params_t;

/* ---- one stream descriptor per upstream ---- */
typedef struct input_stream_t_ {
    stream_kind_t kind;            /* STREAM_PROXIMAL  or  STREAM_DISTAL */
    const u32* activity;           /* packed-32 bitarray, layer reads */
    u32 source_cols;
    u8  source_cells_per_col;      /* 0 or 1 == flat;  2..32 == per-column */
    u32 source_cells;              /* derived; init computes */
    u8  segments_assigned;         /* distal: this stream's quota of segs/cell */
    u8  perm_threshold;            /* p* for connections drawing from this stream */
    u8  activation_threshold;      /* proximal-only: theta_F */
    const char* name;
} input_stream_t;

/* ---- two-phase init ---- */
void layer_init_state(layer_t* L, layer_params_t p);
void layer_init_connections(layer_t* L,
                            const input_stream_t* streams, u8 num_streams,
                            u32* seed);
void layer_init(layer_t* L, layer_params_t p,
                const input_stream_t* streams, u8 num_streams,
                u32* seed);                              /* convenience */
void layer_free(layer_t* L);

/* ---- per step ---- */
void layer_step(layer_t* L, const u32* active_columns_or_NULL);

/* OR break the step into phases (e.g. for L6's grid-shift hook): */
void layer_project(layer_t* L);
void layer_predict(layer_t* L);
void layer_decide (layer_t* L, const u32* active_columns_or_NULL);
void layer_learn  (layer_t* L);

/* L6-style pre-projection hook (toroidal pow-2 shift on active_prev): */
void layer_shift_active_prev_grid(layer_t* L,
                                  vec2d movement, uvec2d log_scale,
                                  u32 log_cols_sqrt);
```

Bitarrays in/out follow the **packed-32 convention** documented at the top of `layer.h`:

- *Per-column layer* (`cells_per_col > 1`): one full `u32` word per column; bit-index of cell `j` in column `c` is `(c << 5) | j`.
- *Flat layer* (`cells_per_col == 1`): standard packed-32 over `cols` bits.

`active_columns` (passed to `layer_decide` for BURST layers) is always packed-32 over `cols` bits regardless of layout.

## 4. Wiring patterns (look at `learning_module.c`)

The repo's [learning_module.c](../src/learning_module.c) is the canonical worked example:

- **Self-recurrence**: a stream points at the *same* layer's `active_prev`. Use the two-phase init (`layer_init_state` → fill streams → `layer_init_connections`).
- **Cross-layer current-step context**: a stream points at another layer's `active`. The convention is that `&L.active` always exposes the latest decision (this is preserved by the snapshot-at-start-of-step pattern in `layer_step`). So a downstream layer's stream pointing at `&upstream.active` correctly reads the same-step decision after the upstream has run.
- **Cross-layer previous-step context**: same as above, but the destination layer runs *before* the source layer this step. The source layer's `active` then still holds the previous decision when read.
- **Top-down feedback**: a `STREAM_DISTAL` whose `activity` points at an upper layer's `active`. No new code path — just an extra entry in the streams array.
- **Cross-LM (`montx-htm`-style)**: same template. The source bitarray is a buffer published by another LM (in this repo, via the `lmat_u32` indirection). v1 of `learning_module.c` carries the parameter through but disables it — wiring it is the obvious next step.

## 5. Pitfalls

1. **`source_cells_per_col` matters**. If your source layer is per-column (e.g. cells laid out as bits inside a u32 per column, with gap bits above `cells_per_col`), set `source_cells_per_col = source.cells_per_col`. If you set `0` or `1`, `init` will draw connection sources from a flat range of size `source_cols`, *not* from `(col, cell)` pairs — most draws will then land in the gap bits of a per-column source and silently produce dead connections. The plumbing tests this in `test_project`.

2. **Phase ordering when layers share state.** Decide upstream-vs-downstream order *before* writing streams. If layer A's distal context comes from layer B at step `t`, run B first, then A. If A reads B at step `t-1`, run A first. The convention `&L.active == latest decision` makes this an explicit knob.

3. **L6-style pre-projection hooks** can't use `layer_step` because the hook (e.g. `layer_shift_active_prev_grid`) must sit *between* the active→active_prev snapshot and `layer_project`. Call the four phase functions manually around the hook (see `learning_module.c:learning_module_step` for the pattern).

4. **Histogram bound in `find_kth_largest_u8`**. The TOPK decision's `kth` finder histograms over `[0, segments_per_cell]`. If you ever stuff `segment_spikes` directly (e.g. in tests), keep values within that range. In normal operation `predict` clamps automatically.

5. **`LAYER_MAX_STREAMS = 8`** is a static cap. Bump it in `layer.h` if your wiring needs more.

## 6. What's left as TODO (open hooks already in the API)

- **Cross-LM external context for L3.** The `lm_segment_split_t.l3_external_segments` field is plumbed; setting it >0 today asserts because v1 only wires the proximal+self-distal streams. To enable it: add a third `STREAM_DISTAL` to L3 in `init_learning_module`, with `activity` pointing at the external LM's published bits.
- **L3 ffwd (proximal) learning.** `enable_ffwd_learning` is honored by `layer_learn` but the proximal update path is currently a no-op. The `ffwd_by_cell_offset / ffwd_by_cell_data` CSR index is built at init, so flipping the flag and writing a small mirror of the distal Hebbian loop is the work.
- **Topology-aware proximal connections.** Today they're a uniform random subset over the source bitarray.
- **Multi-threading.** `project` parallelizes over `active_prev_cells`; `learn` parallelizes over `update_segments`. Both arrays are dense and short.

## 7. Quick start

```bash
# In montx-htm:
cp ../cmontyll-claude/src/{layer,lm_parameters,types,tensor,bitarray,lmat,distributions,algorithms,location}.{h,c} src/   # ignore non-existent .c files (types.h, bitarray.h, location.h are header-only)
cp ../cmontyll-claude/tests/test_layer.c tests/
# Add test target to Makefile (see this repo's Makefile)
make test
```

If `make test` reports `23 passed, 0 failed`, the port is sound. Use `learning_module.c` as a recipe for composing into a full LM.
