#ifndef LAYER_H
#define LAYER_H

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "types.h"
#include "tensor.h"
#include "bitarray.h"
#include "lm_parameters.h"

/* =====================================================================
 * Unified HTM cell-style layer.
 *
 * Realizes the Table-4.2 formulation from Xavier's thesis (§4.1):
 *
 *      mu(G, p*)   synaptic overlap on a connection set G (proximal F or distal D)
 *      tau_ij      = sum_d 1{mu(D_ijd, p*) >= theta_d}                (NMDA spikes)
 *      pi_I        = 1 iff Condition(tau_I)                           (depolarization)
 *      a_I         = Decision(mu(F_I, p*), pi_I^{t-1})                (action potential)
 *      p_{t+1}     = p_t +- delta on the per-step update set X_t      (Hebbian)
 *
 * Layers (Pooler, L4, L6, L3) differ only in (i) which streams are
 * proximal vs distal, (ii) the predicate Condition for pi, (iii) the
 * predicate Decision for a, (iv) which segments end up in X_t.
 *
 * The Pooler stays in pooler.{h,c}: it has no segments and a
 * homeostatic boosting mechanism that does not generalize to L3/L4/L6.
 * ===================================================================== */

/* =====================================================================
 * Bitarray convention
 *
 * ALL activity bitarrays in/out of layer_t are packed-32: bit i lives at
 *   ( activity[i >> 5] >> (i & 31) ) & 1
 *
 * For per-column layers with cells_per_col cells (cells_per_col <= 32),
 * each column is allocated one full u32 word. The bit-index of cell j
 * in column c is therefore  (c << 5) | j  -- equivalently, c * 32 + j.
 * Unused bits are simply never set.
 *
 * For flat layers (e.g. L3 output, lmat external bits), the activity
 * bitarray is just a packed-32 list of cells with bit-index == cell-index.
 *
 * This uniform convention lets a single `source_index` integer address
 * any cell in any source, and lets `GET_BIT_FROM_PACKED32` work
 * everywhere without knowing the source layout.
 * ===================================================================== */

/* =====================================================================
 * Stream descriptor: anything the layer reads from.
 * ===================================================================== */

typedef enum {
    STREAM_PROXIMAL,    /* contributes to a_I via feedforward overlap (theta_F) */
    STREAM_DISTAL       /* contributes to pi_I via NMDA spikes (theta_d) */
} stream_kind_t;

typedef struct input_stream_t_ {
    stream_kind_t kind;

    /* packed-32 bitarray; the layer reads but never writes it. */
    const u32* activity;

    /* Number of valid bit indices; used to size the by-pre index. */
    u32 source_cells;

    /* DISTAL only: how many segments per cell of THIS layer draw their
     * incoming connections from this stream. Must sum across distal
     * streams to layer_params.segments. Ignored for proximal streams. */
    u8 segments_assigned;

    /* p* threshold for connections of THIS layer that come from this
     * stream. A connection contributes to overlap iff permanence >= this. */
    u8 perm_threshold;

    /* PROXIMAL only: theta_F, the minimum proximal overlap required for
     * a cell to qualify as "depolarized via feedforward". Ignored for
     * distal streams. */
    u8 activation_threshold;

    /* Optional debug tag. */
    const char* name;
} input_stream_t;

#define LAYER_MAX_STREAMS 8

/* =====================================================================
 * Decision rule (somatic action potential a_I).
 * ===================================================================== */

typedef enum {
    /* L4 / L6: column-gated. Active columns are passed in via
     * layer_decide(active_columns). Within each active column,
     * predicted cells fire; if no cell is predicted, all cells in the
     * column burst. */
    DECIDE_BURST_OR_PREDICTED,

    /* L3: top-k by tau, gated by feedforward overlap. A cell fires iff
     *   mu(F) >= theta_F  AND  tau_I >= k-th largest tau in the layer.
     * `active_columns` is ignored. The proximal arena must be populated. */
    DECIDE_TOPK_PREDICTED_FFWD
} decision_kind_t;

/* =====================================================================
 * Per-segment metadata that survives phase boundaries.
 * Sized to 4 B for cache density.
 * ===================================================================== */

typedef struct segment_meta_t_ {
    u8 spike_count;     /* mu(D) saved at the end of predict() */
    u8 stream_id;       /* index into layer.streams[] */
    u16 _reserved;
} segment_meta_t;

/* =====================================================================
 * Connection. Stored once in layer.connections; addressed via two CSR
 * indices (by-pre for project, by-seg for learn).
 * Sized to 12 B.
 * ===================================================================== */

typedef struct connection_t_ {
    u32 source_index;   /* bit-index into streams[stream_id].activity */
    u32 segment_index;  /* flat index into segment_accumulators / segment_meta */
    u8  permanence;
    u8  stream_id;
    u16 _reserved;
} connection_t;

/* =====================================================================
 * Update-set tag. Used by learn().
 * ===================================================================== */

typedef enum {
    UPDATE_REINFORCE = 0,   /* +increment on active-prev incidents, -decrement on inactive */
    UPDATE_DECAY     = 1    /* -decay on active-prev incidents that are above threshold */
} update_op_t;

/* =====================================================================
 * Layer parameters.
 * ===================================================================== */

typedef struct layer_params_t_ {
    u16 cols;
    u8  cells_per_col;          /* 1 <= cells_per_col <= 32 */
    u8  segments_per_cell;

    u16 conns_per_segment;      /* uniform fan-in for distal segments */
    u16 ffwd_conns_per_cell;    /* 0 if no proximal stream */

    decision_kind_t decision;
    u16 top_k;                  /* used for DECIDE_TOPK_* */

    htm_params_t htm;           /* shared with legacy code (lm_parameters.h) */

    /* Feature flags. */
    u8 enable_distal_learning;
    u8 enable_decay;
    u8 enable_ffwd_learning;    /* off in v1; placeholder for L3 ffwd updates */
} layer_params_t;

/* =====================================================================
 * The layer.
 * ===================================================================== */

typedef struct layer_t_ {
    layer_params_t p;

    /* ---- Streams (caller-owned bitarrays) ---- */
    input_stream_t streams[LAYER_MAX_STREAMS];
    u8 num_streams;
    /* Cumulative source-cell offsets into by_pre_offset, per stream. */
    u32 stream_src_offsets[LAYER_MAX_STREAMS + 1];
    /* Per-stream contiguous block of distal segments [start, end). */
    u32 stream_seg_offsets[LAYER_MAX_STREAMS + 1];

    /* ---- Distal connection arena, addressed two ways ---- */
    connection_t* connections;          /* [n_distal] */
    u32 n_distal;
    /* by-pre CSR: connection ids grouped by (stream_id, source_cell).
     * by_pre_offset has size (sum over streams of source_cells) + 1.
     * Slice for (s, c): [ by_pre_offset[stream_src_offsets[s] + c],
     *                     by_pre_offset[stream_src_offsets[s] + c + 1] )
     * holds connection ids in by_pre_data. */
    u32* by_pre_offset;
    u32* by_pre_data;
    u32  by_pre_offset_len;             /* = stream_src_offsets[num_streams] + 1 */
    /* by-seg CSR: connection ids grouped by post segment index.
     * Slice for segment s: [ by_seg_offset[s], by_seg_offset[s+1] ) */
    u32* by_seg_offset;                 /* size = total_segments + 1 */
    u32* by_seg_data;                   /* size = n_distal */

    /* ---- Proximal arena (one stream max in v1) ---- */
    connection_t* ffwd_connections;     /* [n_ffwd] */
    u32 n_ffwd;
    int proximal_stream_id;             /* -1 if no proximal stream */
    u32* ffwd_by_pre_offset;            /* size = streams[proximal].source_cells + 1 */
    u32* ffwd_by_pre_data;              /* size = n_ffwd */
    u32* ffwd_by_cell_offset;           /* size = total_cells + 1 (for ffwd learn, future) */
    u32* ffwd_by_cell_data;             /* size = n_ffwd */

    /* ---- Per-segment dynamic state ---- */
    tnsr_u8 segment_accumulators;       /* [cols, cells, segments]; cleared in project */
    segment_meta_t* segment_meta;       /* [total_segments] */

    /* ---- Per-cell scratch (small, dense) ---- */
    u8* ffwd_overlap;                   /* [total_cells] mu(F), produced in decide() */
    u8* segment_spikes;                 /* [total_cells] tau_I = #spiking segments */

    /* ---- Activity bitarrays (output / state) ---- */
    /* For per-column layers: one u32 word per column (cells laid out as bits).
     * For flat layers (cols=N, cells_per_col=1): packed-32 bitarray of N bits. */
    u32* active;
    u32* predicted;
    u32* active_prev;
    u32  activity_words;                /* size in u32 of each activity buffer */

    /* ---- Sparse active-cell index ---- *
     * Built at end of decide(); consumed by next-step project() and by learn().
     * Each entry is the bit-index used in the activity bitarray:
     *   per-column layers:  (col << 5) | cell
     *   flat layers:        cell
     */
    u32* active_cells;
    u32  active_cells_count;
    u32* active_prev_cells;
    u32  active_prev_cells_count;

    /* ---- Update-set buffers (built between decide and learn) ---- */
    u32* update_segments;               /* flat segment indices */
    u8*  update_segment_op;             /* update_op_t, one per update segment */
    u32  update_segments_count;
    u32  update_segments_capacity;

    /* ---- Cached counts to avoid recomputation ---- */
    u32 total_cells;                    /* cols * cells_per_col */
    u32 total_segments;                 /* total_cells * segments_per_cell */
} layer_t;

/* =====================================================================
 * Lifecycle
 * ===================================================================== */

/* Two-phase init.
 *
 * `layer_init_state` allocates the activity bitarrays (active, predicted,
 * active_prev) and per-cell / per-segment scratch. It does *not* require
 * stream pointers, which lets the caller, after this call, point a
 * self-recurrent stream's `activity` at `&L->active_prev[0]`. Then call
 * `layer_init_connections` with the now-fully-wired streams.
 *
 * The caller's bitarrays must remain valid until `layer_free`. `seed` is
 * consumed by `unif_rand_range_u32`. */
void layer_init_state(layer_t* L, layer_params_t p);
void layer_init_connections(layer_t* L,
                            const input_stream_t* streams, u8 num_streams,
                            u32* seed);

/* Convenience wrapper for non-recurrent layers: state then connections. */
void layer_init(layer_t* L, layer_params_t p,
                const input_stream_t* streams, u8 num_streams,
                u32* seed);

void layer_free(layer_t* L);

/* =====================================================================
 * Phases. Call in order, or use layer_step() for the canonical flow.
 * ===================================================================== */

/* Distal projection: clear segment_accumulators, then for each (stream,
 * source_cell) where the source bit is set, increment the target
 * accumulator of every connection whose permanence >= stream.perm_threshold.
 * Saturates accumulators at 255. */
void layer_project(layer_t* L);

/* Read accumulators -> segment_meta.spike_count, compute tau per cell
 * -> segment_spikes. Set the `predicted` bitarray as the layer's
 * Condition(tau): tau >= htm.predicted_threshold for BURST decisions;
 * for TOPK the predicate is deferred to layer_decide. */
void layer_predict(layer_t* L);

/* Compute the somatic action potential a_I and write `active` +
 * `active_cells`. For DECIDE_BURST_OR_PREDICTED, `active_columns` is a
 * required packed-32 bitarray of length `cols` words selecting which
 * columns are gated open by the upstream pooler/movement. For
 * DECIDE_TOPK_PREDICTED_FFWD, `active_columns` is ignored and may be NULL. */
void layer_decide(layer_t* L, const u32* active_columns);

/* Build the update set from the just-decided activations and apply the
 * Hebbian rules over the by-seg index. Reads streams[s].activity at
 * the time learn() runs; the caller must arrange it to reflect the
 * incident-prev semantics expected by the thesis (in practice: during
 * a step, sources keep their previous-step activity until they
 * themselves run). */
void layer_learn(layer_t* L);

/* project + predict + decide + learn, then swap (active, active_cells)
 * <-> (active_prev, active_prev_cells). */
void layer_step(layer_t* L, const u32* active_columns);

/* =====================================================================
 * Optional pre-projection hook (used by L6 grid-cell layer)
 *
 * Permutes active_prev / active_prev_cells along a 2D movement vector,
 * matching the toroidal grid shift in the legacy location_layer
 * (location_layer.c). Must be called between the previous step's
 * swap and this step's project().
 * ===================================================================== */
void layer_shift_active_prev_grid(layer_t* L,
                                  vec2d movement,
                                  uvec2d log_scale,
                                  u32 log_cols_sqrt);

/* =====================================================================
 * Introspection
 * ===================================================================== */

u32  layer_connections_footprint_bytes(const layer_params_t* p, u8 num_streams);
u32  layer_state_footprint_bytes(const layer_params_t* p);
void layer_print_memory_footprint(const layer_t* L);

/* =====================================================================
 * Helpers / inlines
 * ===================================================================== */

/* Bit-test on a packed-32 bitarray, robust against the legacy macro's
 * unparenthesized expansion when used in expressions. */
static inline u32 layer_bit_get(const u32* packed, u32 idx) {
    return (packed[idx >> 5] >> (idx & 31u)) & 1u;
}

static inline void layer_bit_set(u32* packed, u32 idx) {
    packed[idx >> 5] |= (1u << (idx & 31u));
}

#endif /* LAYER_H */
