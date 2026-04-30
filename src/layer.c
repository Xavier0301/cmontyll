/* =====================================================================
 * Unified HTM cell-style layer -- implementation.
 *
 * See layer.h and docs/htm_unified.md for the data model and the
 * Table-4.2 mapping. This file owns five phase functions:
 *
 *   layer_project   sparse: active_prev_cells x by_pre  -> accumulators
 *   layer_predict   dense:  accumulators -> spike_count, tau, predicted
 *   layer_decide    BURST or TOPK: writes active + active_cells
 *   layer_learn     sparse: update_set x by_seg -> permanence deltas
 *   layer_step      glue + active <-> active_prev swap
 *
 * plus a pre-projection hook (layer_shift_active_prev_grid) for the
 * L6 grid layer.
 * ===================================================================== */

#include "layer.h"

#include <assert.h>

#include "distributions.h"

/* =====================================================================
 * Compile-time / private constants
 * ===================================================================== */

#ifndef LAYER_DEBUG
#define LAYER_DEBUG 0
#endif

/* =====================================================================
 * Private helpers
 * ===================================================================== */

static inline u32 cell_global_index(const layer_t* L, u32 col, u32 cell) {
    return col * L->p.cells_per_col + cell;
}

static inline u32 segment_global_index(const layer_t* L,
                                       u32 col, u32 cell, u32 seg_in_cell) {
    return cell_global_index(L, col, cell) * L->p.segments_per_cell + seg_in_cell;
}

/* Decode a packed sparse cell index to (col, cell).
 *   Per-column layers: bit-index encoding is (col << 5) | cell.
 *   Flat layers (cells_per_col == 1): bit-index == col, cell == 0. */
static inline void decode_active_cell(const layer_t* L, u32 idx,
                                      u32* out_col, u32* out_cell) {
    if (L->p.cells_per_col == 1) {
        *out_col = idx;
        *out_cell = 0;
    } else {
        *out_col = idx >> 5;
        *out_cell = idx & 31u;
    }
}

/* Encode (col, cell) to packed bit-index. */
static inline u32 encode_active_cell(const layer_t* L, u32 col, u32 cell) {
    return (L->p.cells_per_col == 1) ? col : ((col << 5) | cell);
}

/* Effective cells-per-col of a stream's source layout (>= 1). */
static inline u8 stream_eff_sccpc(const input_stream_t* s) {
    return (s->source_cells_per_col <= 1) ? 1 : s->source_cells_per_col;
}

/* Encode (src_col, src_cell) of a stream into the bit-index used by
 * stream.activity. */
static inline u32 stream_bit_index(const input_stream_t* s,
                                   u32 src_col, u32 src_cell) {
    return (s->source_cells_per_col <= 1) ? src_col
                                          : ((src_col << 5) | src_cell);
}

/* Compact bucket id of (src_col, src_cell) inside one stream. Total
 * buckets per stream = source_cols * eff_sccpc. */
static inline u32 stream_compact_id(const input_stream_t* s,
                                    u32 src_col, u32 src_cell) {
    return src_col * stream_eff_sccpc(s) + src_cell;
}

/* Stream id of the local segment slot `seg_in_cell` within a cell. */
static u8 stream_for_local_segment(const layer_t* L, u32 seg_in_cell) {
    for (u8 s = 0; s < L->num_streams; ++s) {
        if (L->streams[s].kind != STREAM_DISTAL) continue;
        if (seg_in_cell >= L->stream_seg_offsets[s] &&
            seg_in_cell <  L->stream_seg_offsets[s + 1]) {
            return s;
        }
    }
    /* unreachable if init invariants hold */
    return 0xFF;
}

/* CSR builder: given a flat array `keys[n]` of bucket ids in
 * [0, num_buckets), produces:
 *   offset[num_buckets + 1] = cumulative bucket sizes
 *   data[n] = connection ids sorted by bucket (stable)
 * Uses two passes (count then scatter). */
static void csr_build(u32 num_buckets, u32 n,
                      const u32* keys,
                      u32* offset, u32* data) {
    for (u32 i = 0; i <= num_buckets; ++i) offset[i] = 0;
    for (u32 i = 0; i < n; ++i) {
        u32 k = keys[i];
        ++offset[k + 1];
    }
    for (u32 i = 1; i <= num_buckets; ++i) offset[i] += offset[i - 1];
    /* scratch cursor: copy of offsets we will advance */
    u32* cursor = (u32*) malloc((num_buckets + 1) * sizeof(u32));
    for (u32 i = 0; i <= num_buckets; ++i) cursor[i] = offset[i];
    for (u32 i = 0; i < n; ++i) {
        u32 k = keys[i];
        data[cursor[k]++] = i;
    }
    free(cursor);
}

/* =====================================================================
 * Lifecycle: state allocation
 * ===================================================================== */

void layer_init_state(layer_t* L, layer_params_t p) {
    memset(L, 0, sizeof(*L));
    L->p = p;
    L->total_cells = (u32) p.cols * p.cells_per_col;
    L->total_segments = L->total_cells * p.segments_per_cell;
    L->proximal_stream_id = -1;

    /* Activity bitarrays: per-column layers use one u32 per col;
     * flat layers use ceil(cols/32). */
    if (p.cells_per_col == 1) {
        L->activity_words = (p.cols + 31u) >> 5;
    } else {
        L->activity_words = p.cols;
    }
    u32 act_bytes = L->activity_words * sizeof(u32);
    L->active      = (u32*) calloc(L->activity_words, sizeof(u32));
    L->predicted   = (u32*) calloc(L->activity_words, sizeof(u32));
    L->active_prev = (u32*) calloc(L->activity_words, sizeof(u32));
    (void) act_bytes;

    /* Per-segment scratch and metadata */
    tnsr_u8_init(&L->segment_accumulators,
                 p.cols, p.cells_per_col, p.segments_per_cell);
    memset(L->segment_accumulators.data, 0,
           L->total_segments * sizeof(u8));

    L->segment_meta = (segment_meta_t*) calloc(L->total_segments,
                                               sizeof(segment_meta_t));

    /* Per-cell scratch */
    L->ffwd_overlap   = (u8*) calloc(L->total_cells, sizeof(u8));
    L->segment_spikes = (u8*) calloc(L->total_cells, sizeof(u8));

    /* Sparse active-cell lists. Worst case = all cells active. */
    L->active_cells       = (u32*) malloc(L->total_cells * sizeof(u32));
    L->active_prev_cells  = (u32*) malloc(L->total_cells * sizeof(u32));
    L->active_cells_count = 0;
    L->active_prev_cells_count = 0;

    /* Update-set buffers. Cap at total_segments which is a safe upper bound
     * (one entry per segment per step in the worst case). */
    L->update_segments_capacity = L->total_segments;
    L->update_segments    = (u32*) malloc(L->update_segments_capacity * sizeof(u32));
    L->update_segment_op  = (u8*)  malloc(L->update_segments_capacity * sizeof(u8));
    L->update_segments_count = 0;
}

/* =====================================================================
 * Lifecycle: connection arena + CSR indices
 * ===================================================================== */

void layer_init_connections(layer_t* L,
                            const input_stream_t* streams, u8 num_streams,
                            u32* seed) {
    assert(num_streams <= LAYER_MAX_STREAMS);
    L->num_streams = num_streams;
    for (u8 s = 0; s < num_streams; ++s) {
        L->streams[s] = streams[s];
        /* Derive source_cells from source_cols * eff_sccpc. */
        u8 sccpc = stream_eff_sccpc(&L->streams[s]);
        L->streams[s].source_cells = L->streams[s].source_cols * sccpc;
    }

    /* ---- Validate streams; compute per-stream offsets ----
     * Distal: segments_assigned must sum to segments_per_cell.
     * Proximal: at most one in v1; ffwd_conns_per_cell decides arena size. */
    u32 distal_segs_per_cell = 0;
    int proximal_id = -1;
    for (u8 s = 0; s < num_streams; ++s) {
        if (L->streams[s].kind == STREAM_DISTAL) {
            distal_segs_per_cell += L->streams[s].segments_assigned;
        } else {
            assert(proximal_id < 0 && "v1: at most one proximal stream");
            proximal_id = s;
        }
    }
    assert(distal_segs_per_cell == L->p.segments_per_cell);
    (void) distal_segs_per_cell;
    L->proximal_stream_id = proximal_id;

    /* Per-stream segment offsets within a cell (distal only). */
    {
        u32 acc = 0;
        for (u8 s = 0; s < num_streams; ++s) {
            L->stream_seg_offsets[s] = acc;
            if (L->streams[s].kind == STREAM_DISTAL) {
                acc += L->streams[s].segments_assigned;
            }
        }
        L->stream_seg_offsets[num_streams] = acc;
    }

    /* Per-stream compact-source offsets (used as CSR bucket-id base for
     * by_pre). Number of compact buckets per stream = source_cols * eff_sccpc. */
    {
        u32 acc = 0;
        for (u8 s = 0; s < num_streams; ++s) {
            L->stream_src_offsets[s] = acc;
            if (L->streams[s].kind == STREAM_DISTAL) {
                acc += L->streams[s].source_cols * stream_eff_sccpc(&L->streams[s]);
            }
        }
        L->stream_src_offsets[num_streams] = acc;
        L->by_pre_offset_len = acc + 1;
    }

    /* ---- Distal arena ---- */
    L->n_distal = L->total_segments * L->p.conns_per_segment;
    L->connections = (connection_t*) malloc(L->n_distal * sizeof(connection_t));

    /* segment_meta.stream_id is fully determined by local segment index
     * within a cell. Pre-fill once. */
    for (u32 col = 0; col < L->p.cols; ++col) {
        for (u32 cell = 0; cell < L->p.cells_per_col; ++cell) {
            for (u32 seg = 0; seg < L->p.segments_per_cell; ++seg) {
                u32 g = segment_global_index(L, col, cell, seg);
                L->segment_meta[g].spike_count = 0;
                L->segment_meta[g].stream_id = stream_for_local_segment(L, seg);
                L->segment_meta[g]._reserved = 0;
            }
        }
    }

    /* Draw distal connections. For each (col, cell, local_seg, k) pick a
     * source (src_col, src_cell) from that local_seg's stream,
     * randomize permanence. Encoding the bit-index from (src_col, src_cell)
     * guarantees we never land in the gap bits of a per-column source. */
    u32 ci = 0;
    for (u32 col = 0; col < L->p.cols; ++col) {
        for (u32 cell = 0; cell < L->p.cells_per_col; ++cell) {
            for (u32 seg = 0; seg < L->p.segments_per_cell; ++seg) {
                u8 sid = stream_for_local_segment(L, seg);
                const input_stream_t* str = &L->streams[sid];
                u8 sccpc = stream_eff_sccpc(str);
                u32 g = segment_global_index(L, col, cell, seg);
                for (u32 k = 0; k < L->p.conns_per_segment; ++k) {
                    connection_t* c = &L->connections[ci++];
                    u32 src_col  = unif_rand_range_u32(0, str->source_cols - 1, seed);
                    u32 src_cell = (sccpc == 1) ? 0u
                                                : unif_rand_range_u32(0, sccpc - 1, seed);
                    c->source_index = stream_bit_index(str, src_col, src_cell);
                    c->segment_index = g;
                    c->permanence = (u8) unif_rand_range_u32(0, 255, seed);
                    c->stream_id = sid;
                    c->_reserved = 0;
                }
            }
        }
    }
    assert(ci == L->n_distal);

    /* ---- by_seg index ---- */
    L->by_seg_offset = (u32*) malloc((L->total_segments + 1) * sizeof(u32));
    L->by_seg_data   = (u32*) malloc(L->n_distal * sizeof(u32));
    {
        u32* keys = (u32*) malloc(L->n_distal * sizeof(u32));
        for (u32 i = 0; i < L->n_distal; ++i)
            keys[i] = L->connections[i].segment_index;
        csr_build(L->total_segments, L->n_distal, keys,
                  L->by_seg_offset, L->by_seg_data);
        free(keys);
    }

    /* ---- by_pre index ---- *
     * Buckets indexed by (stream_id, src_col, src_cell) -> compact id =
     * stream_src_offsets[s] + src_col * eff_sccpc + src_cell. We recover
     * the compact (col, cell) from connection.source_index (which is the
     * activity bit-index). */
    u32 num_pre_buckets = L->stream_src_offsets[num_streams];
    L->by_pre_offset = (u32*) malloc((num_pre_buckets + 1) * sizeof(u32));
    L->by_pre_data   = (u32*) malloc(L->n_distal * sizeof(u32));
    {
        u32* keys = (u32*) malloc(L->n_distal * sizeof(u32));
        for (u32 i = 0; i < L->n_distal; ++i) {
            const connection_t* c = &L->connections[i];
            const input_stream_t* str = &L->streams[c->stream_id];
            u8 sccpc = stream_eff_sccpc(str);
            u32 src_col, src_cell;
            if (sccpc == 1) {
                src_col = c->source_index; src_cell = 0;
            } else {
                src_col = c->source_index >> 5;
                src_cell = c->source_index & 31u;
            }
            keys[i] = L->stream_src_offsets[c->stream_id]
                    + stream_compact_id(str, src_col, src_cell);
        }
        csr_build(num_pre_buckets, L->n_distal, keys,
                  L->by_pre_offset, L->by_pre_data);
        free(keys);
    }

    /* ---- Proximal arena ---- */
    if (proximal_id >= 0 && L->p.ffwd_conns_per_cell > 0) {
        const input_stream_t* prox = &L->streams[proximal_id];
        L->n_ffwd = L->total_cells * L->p.ffwd_conns_per_cell;
        L->ffwd_connections =
            (connection_t*) malloc(L->n_ffwd * sizeof(connection_t));

        /* Each ffwd connection is stored with segment_index = target cell
         * global index (we treat the cell as a single virtual segment for
         * indexing purposes; the by-cell index plays the role of by-seg).
         * source_index encodes (src_col, src_cell) for the proximal stream. */
        u8 prox_sccpc = stream_eff_sccpc(prox);
        u32 fci = 0;
        for (u32 col = 0; col < L->p.cols; ++col) {
            for (u32 cell = 0; cell < L->p.cells_per_col; ++cell) {
                u32 cg = cell_global_index(L, col, cell);
                for (u32 k = 0; k < L->p.ffwd_conns_per_cell; ++k) {
                    connection_t* c = &L->ffwd_connections[fci++];
                    u32 src_col  = unif_rand_range_u32(0, prox->source_cols - 1, seed);
                    u32 src_cell = (prox_sccpc == 1) ? 0u
                                                     : unif_rand_range_u32(0, prox_sccpc - 1, seed);
                    c->source_index = stream_bit_index(prox, src_col, src_cell);
                    c->segment_index = cg;
                    c->permanence = (u8) unif_rand_range_u32(0, 255, seed);
                    c->stream_id = (u8) proximal_id;
                    c->_reserved = 0;
                }
            }
        }
        assert(fci == L->n_ffwd);

        /* by_cell index (== by-seg analogue for ffwd) */
        L->ffwd_by_cell_offset =
            (u32*) malloc((L->total_cells + 1) * sizeof(u32));
        L->ffwd_by_cell_data = (u32*) malloc(L->n_ffwd * sizeof(u32));
        {
            u32* keys = (u32*) malloc(L->n_ffwd * sizeof(u32));
            for (u32 i = 0; i < L->n_ffwd; ++i)
                keys[i] = L->ffwd_connections[i].segment_index;
            csr_build(L->total_cells, L->n_ffwd, keys,
                      L->ffwd_by_cell_offset, L->ffwd_by_cell_data);
            free(keys);
        }

        /* by_pre index for ffwd: buckets keyed by compact id over the
         * proximal stream (src_col * eff_sccpc + src_cell). */
        u32 prox_buckets = prox->source_cols * prox_sccpc;
        L->ffwd_by_pre_offset =
            (u32*) malloc((prox_buckets + 1) * sizeof(u32));
        L->ffwd_by_pre_data = (u32*) malloc(L->n_ffwd * sizeof(u32));
        {
            u32* keys = (u32*) malloc(L->n_ffwd * sizeof(u32));
            for (u32 i = 0; i < L->n_ffwd; ++i) {
                u32 si = L->ffwd_connections[i].source_index;
                u32 src_col, src_cell;
                if (prox_sccpc == 1) { src_col = si; src_cell = 0; }
                else { src_col = si >> 5; src_cell = si & 31u; }
                keys[i] = stream_compact_id(prox, src_col, src_cell);
            }
            csr_build(prox_buckets, L->n_ffwd, keys,
                      L->ffwd_by_pre_offset, L->ffwd_by_pre_data);
            free(keys);
        }
    } else {
        L->n_ffwd = 0;
        L->ffwd_connections = NULL;
        L->ffwd_by_pre_offset = NULL;
        L->ffwd_by_pre_data = NULL;
        L->ffwd_by_cell_offset = NULL;
        L->ffwd_by_cell_data = NULL;
    }
}

void layer_init(layer_t* L, layer_params_t p,
                const input_stream_t* streams, u8 num_streams,
                u32* seed) {
    layer_init_state(L, p);
    layer_init_connections(L, streams, num_streams, seed);
}

void layer_free(layer_t* L) {
    free(L->active);
    free(L->predicted);
    free(L->active_prev);

    free(L->segment_accumulators.data);
    free(L->segment_meta);
    free(L->ffwd_overlap);
    free(L->segment_spikes);

    free(L->active_cells);
    free(L->active_prev_cells);
    free(L->update_segments);
    free(L->update_segment_op);

    free(L->connections);
    free(L->by_pre_offset);
    free(L->by_pre_data);
    free(L->by_seg_offset);
    free(L->by_seg_data);

    free(L->ffwd_connections);
    free(L->ffwd_by_pre_offset);
    free(L->ffwd_by_pre_data);
    free(L->ffwd_by_cell_offset);
    free(L->ffwd_by_cell_data);

    memset(L, 0, sizeof(*L));
}

/* =====================================================================
 * layer_project
 *   Sparse iteration over active_prev_cells of each distal stream, then
 *   over the by_pre bucket for that source. Saturates accumulators at 255.
 *
 * The layer's *own* recurrent stream (when streams[s].activity points at
 * &L->active_prev[0]) is handled exactly the same way as any other
 * stream -- we walk its activity bitarray densely up to source_cells.
 * For non-self streams we also walk densely; bit-test is fast and the
 * 98% no-op skip keeps cache pressure low. (A future optimization is
 * for the caller to publish a sparse list per stream too.)
 * ===================================================================== */

void layer_project(layer_t* L) {
    /* clear accumulators */
    memset(L->segment_accumulators.data, 0, L->total_segments * sizeof(u8));

    for (u8 s = 0; s < L->num_streams; ++s) {
        const input_stream_t* str = &L->streams[s];
        if (str->kind != STREAM_DISTAL) continue;
        if (str->activity == NULL) continue;

        u32 base = L->stream_src_offsets[s];
        u8 sccpc = stream_eff_sccpc(str);

        if (sccpc == 1) {
            /* flat source: bit-index == compact id == src_col */
            for (u32 src = 0; src < str->source_cols; ++src) {
                if (!layer_bit_get(str->activity, src)) continue;
                u32 a = L->by_pre_offset[base + src];
                u32 b = L->by_pre_offset[base + src + 1];
                for (u32 k = a; k < b; ++k) {
                    const connection_t* c = &L->connections[L->by_pre_data[k]];
                    if (c->permanence < str->perm_threshold) continue;
                    u8* acc = &L->segment_accumulators.data[c->segment_index];
                    if (*acc < 255) ++(*acc);
                }
            }
        } else {
            /* per-column source: walk one word at a time, bit-extract
             * cells that fit in the layer's cells-per-col. */
            for (u32 src_col = 0; src_col < str->source_cols; ++src_col) {
                u32 word = str->activity[src_col];
                if (!word) continue;
                u32 limit = (sccpc >= 32) ? 0xFFFFFFFFu : ((1u << sccpc) - 1u);
                u32 w = word & limit;
                while (w) {
                    u32 src_cell = __builtin_ctz(w);
                    w &= w - 1u;
                    u32 compact = src_col * sccpc + src_cell;
                    u32 a = L->by_pre_offset[base + compact];
                    u32 b = L->by_pre_offset[base + compact + 1];
                    for (u32 k = a; k < b; ++k) {
                        const connection_t* c = &L->connections[L->by_pre_data[k]];
                        if (c->permanence < str->perm_threshold) continue;
                        u8* acc = &L->segment_accumulators.data[c->segment_index];
                        if (*acc < 255) ++(*acc);
                    }
                }
            }
        }
    }
}

/* =====================================================================
 * layer_predict
 *   Per cell, count spiking segments (mu >= theta_d). Save spike_count
 *   into segment_meta. Set predicted bit when tau_I >= predicted_threshold.
 * ===================================================================== */

void layer_predict(layer_t* L) {
    const u8 theta_d = L->p.htm.segment_spiking_threshold;
    const u8 pred_t  = L->p.htm.predicted_threshold;

    /* zero predicted */
    memset(L->predicted, 0, L->activity_words * sizeof(u32));

    const u8* acc = L->segment_accumulators.data;

    for (u32 col = 0; col < L->p.cols; ++col) {
        u32 pred_word = 0;
        for (u32 cell = 0; cell < L->p.cells_per_col; ++cell) {
            u32 tau = 0;
            for (u32 seg = 0; seg < L->p.segments_per_cell; ++seg) {
                u8 v = *acc++;
                u32 g = segment_global_index(L, col, cell, seg);
                L->segment_meta[g].spike_count = v;
                if (v >= theta_d) ++tau;
            }
            if (tau > 255) tau = 255;
            L->segment_spikes[cell_global_index(L, col, cell)] = (u8) tau;
            if (tau >= pred_t) {
                u32 idx = encode_active_cell(L, col, cell);
                pred_word |= (1u << (idx & 31u));
            }
        }
        if (L->p.cells_per_col == 1) {
            /* flat layer: write into the right word */
            if (pred_word) layer_bit_set(L->predicted, col);
        } else {
            L->predicted[col] = pred_word;
        }
    }
}

/* =====================================================================
 * layer_decide
 *
 * Two cases:
 *   DECIDE_BURST_OR_PREDICTED  (L4, L6)
 *     For each col gated open by `active_columns`:
 *       if any cell predicted -> those cells active
 *       else -> all cells in the col burst
 *
 *   DECIDE_TOPK_PREDICTED_FFWD (L3)
 *     Compute proximal mu(F) per cell via sparse iteration over the
 *     proximal stream's active source cells. Find k-th largest tau_I
 *     via a small histogram. A cell is active iff
 *     mu(F) >= theta_F  AND  tau_I >= kth_largest.
 * ===================================================================== */

static void compute_ffwd_overlap_sparse(layer_t* L) {
    memset(L->ffwd_overlap, 0, L->total_cells * sizeof(u8));
    if (L->proximal_stream_id < 0) return;

    const input_stream_t* prox = &L->streams[L->proximal_stream_id];
    if (prox->activity == NULL) return;
    u8 sccpc = stream_eff_sccpc(prox);

    if (sccpc == 1) {
        for (u32 src = 0; src < prox->source_cols; ++src) {
            if (!layer_bit_get(prox->activity, src)) continue;
            u32 a = L->ffwd_by_pre_offset[src];
            u32 b = L->ffwd_by_pre_offset[src + 1];
            for (u32 k = a; k < b; ++k) {
                const connection_t* c = &L->ffwd_connections[L->ffwd_by_pre_data[k]];
                if (c->permanence < prox->perm_threshold) continue;
                u8* o = &L->ffwd_overlap[c->segment_index];
                if (*o < 255) ++(*o);
            }
        }
    } else {
        for (u32 src_col = 0; src_col < prox->source_cols; ++src_col) {
            u32 word = prox->activity[src_col];
            if (!word) continue;
            u32 limit = (sccpc >= 32) ? 0xFFFFFFFFu : ((1u << sccpc) - 1u);
            u32 w = word & limit;
            while (w) {
                u32 src_cell = __builtin_ctz(w);
                w &= w - 1u;
                u32 compact = src_col * sccpc + src_cell;
                u32 a = L->ffwd_by_pre_offset[compact];
                u32 b = L->ffwd_by_pre_offset[compact + 1];
                for (u32 k = a; k < b; ++k) {
                    const connection_t* c = &L->ffwd_connections[L->ffwd_by_pre_data[k]];
                    if (c->permanence < prox->perm_threshold) continue;
                    u8* o = &L->ffwd_overlap[c->segment_index];
                    if (*o < 255) ++(*o);
                }
            }
        }
    }
}

/* Counting-sort find-k-th-largest over a small score range [0, max_score]. */
static u8 find_kth_largest_u8(const u8* scores, u32 n, u32 max_score, u32 k) {
    if (k == 0 || n == 0) return 0;
    /* histogram fits on the stack: max_score is segments_per_cell (u8). */
    u32 hist[256] = {0};
    if (max_score > 255) max_score = 255;
    for (u32 i = 0; i < n; ++i) ++hist[scores[i]];
    u32 seen = 0;
    for (i32 v = (i32) max_score; v >= 0; --v) {
        seen += hist[v];
        if (seen >= k) return (u8) v;
    }
    return 0;
}

void layer_decide(layer_t* L, const u32* active_columns) {
    L->active_cells_count = 0;
    memset(L->active, 0, L->activity_words * sizeof(u32));

    if (L->p.decision == DECIDE_BURST_OR_PREDICTED) {
        assert(active_columns != NULL && "BURST decision requires active_columns");
        u32 cell_mask = (L->p.cells_per_col >= 32)
                          ? 0xFFFFFFFFu
                          : ((1u << L->p.cells_per_col) - 1u);

        for (u32 col = 0; col < L->p.cols; ++col) {
            if (!layer_bit_get(active_columns, col)) continue;

            u32 pred = (L->p.cells_per_col == 1)
                ? layer_bit_get(L->predicted, col)
                : (L->predicted[col] & cell_mask);
            u32 act_word;
            if (pred != 0) {
                act_word = pred;
            } else {
                act_word = cell_mask; /* burst all cells */
            }

            if (L->p.cells_per_col == 1) {
                if (act_word) {
                    layer_bit_set(L->active, col);
                    L->active_cells[L->active_cells_count++] = col;
                }
            } else {
                L->active[col] = act_word;
                u32 w = act_word;
                while (w) {
                    u32 b = __builtin_ctz(w);
                    L->active_cells[L->active_cells_count++] =
                        encode_active_cell(L, col, b);
                    w &= w - 1u;
                }
            }
        }
        return;
    }

    /* DECIDE_TOPK_PREDICTED_FFWD */
    compute_ffwd_overlap_sparse(L);

    u32 k = L->p.top_k;
    u8 kth = find_kth_largest_u8(L->segment_spikes,
                                 L->total_cells,
                                 L->p.segments_per_cell,
                                 k);
    /* threshold_F: pick the proximal stream's activation_threshold
     * (defaults to 0 when no stream was tagged). */
    u8 theta_F = 0;
    if (L->proximal_stream_id >= 0) {
        theta_F = L->streams[L->proximal_stream_id].activation_threshold;
    }

    for (u32 col = 0; col < L->p.cols; ++col) {
        u32 act_word = 0;
        for (u32 cell = 0; cell < L->p.cells_per_col; ++cell) {
            u32 cg = cell_global_index(L, col, cell);
            u8 tau = L->segment_spikes[cg];
            u8 fov = L->ffwd_overlap[cg];
            if (fov >= theta_F && tau >= kth && tau > 0) {
                u32 idx = encode_active_cell(L, col, cell);
                act_word |= (1u << (idx & 31u));
                L->active_cells[L->active_cells_count++] = idx;
            }
        }
        if (L->p.cells_per_col == 1) {
            if (act_word) layer_bit_set(L->active, col);
        } else {
            L->active[col] = act_word;
        }
    }
}

/* =====================================================================
 * layer_learn
 *
 * Step 1: build the update set
 *   REINFORCE
 *     - every spiking segment of every (active && predicted) cell
 *     - the best-match segment of the winner cell in every bursting
 *       column (BURST decision only)
 *   DECAY  (when enable_decay)
 *     - every spiking segment of every (predicted && !active) cell
 *
 * Step 2: walk update_segments x by_seg, apply +=increment / -=decrement
 * (REINFORCE) or saturating decay (DECAY).
 *
 * Note (winner-cell metric): we use sum of segment_meta.spike_count
 * across the cell's segments as the winner score. spike_count is
 * mu(D, p*) -- counts only above-threshold connections. A simplification
 * vs the legacy raw-connection-count proxy, but matches the unified
 * Table 4.2 formulation.
 * ===================================================================== */

static void push_update(layer_t* L, u32 seg, u8 op) {
    if (L->update_segments_count >= L->update_segments_capacity) return;
    L->update_segments[L->update_segments_count] = seg;
    L->update_segment_op[L->update_segments_count] = op;
    ++L->update_segments_count;
}

static void learn_collect_burst(layer_t* L) {
    const u8 theta_d = L->p.htm.segment_spiking_threshold;
    const u8 cells_per_col = L->p.cells_per_col;
    const u8 segs = L->p.segments_per_cell;
    u32 cell_mask = (cells_per_col >= 32) ? 0xFFFFFFFFu
                                          : ((1u << cells_per_col) - 1u);

    for (u32 col = 0; col < L->p.cols; ++col) {
        u32 act_word = (cells_per_col == 1)
            ? (layer_bit_get(L->active, col) ? 1u : 0u)
            : (L->active[col] & cell_mask);
        u32 pred_word = (cells_per_col == 1)
            ? (layer_bit_get(L->predicted, col) ? 1u : 0u)
            : (L->predicted[col] & cell_mask);

        /* REINFORCE / bursting paths only fire on active columns. */
        if (act_word != 0 && pred_word != 0) {
            /* correctly predicted cells: REINFORCE every spiking segment
             * (active AND predicted bits both set) */
            u32 ap = act_word & pred_word;
            while (ap) {
                u32 cell = __builtin_ctz(ap);
                ap &= ap - 1u;
                for (u32 seg = 0; seg < segs; ++seg) {
                    u32 g = segment_global_index(L, col, cell, seg);
                    if (L->segment_meta[g].spike_count >= theta_d) {
                        push_update(L, g, UPDATE_REINFORCE);
                    }
                }
            }
        } else if (act_word != 0) {
            /* bursting column: pick winner cell + best segment */
            u32 winner_cell = 0;
            u32 winner_score = 0;
            for (u32 cell = 0; cell < cells_per_col; ++cell) {
                u32 score = 0;
                for (u32 seg = 0; seg < segs; ++seg) {
                    u32 g = segment_global_index(L, col, cell, seg);
                    score += L->segment_meta[g].spike_count;
                }
                if (score > winner_score) {
                    winner_score = score;
                    winner_cell = cell;
                }
            }
            u32 best_seg = 0;
            u8 best_score = 0;
            for (u32 seg = 0; seg < segs; ++seg) {
                u32 g = segment_global_index(L, col, winner_cell, seg);
                if (L->segment_meta[g].spike_count >= best_score) {
                    best_score = L->segment_meta[g].spike_count;
                    best_seg = seg;
                }
            }
            u32 g = segment_global_index(L, col, winner_cell, best_seg);
            push_update(L, g, UPDATE_REINFORCE);
        }

        /* DECAY on predicted-but-inactive cells. Applies regardless of
         * whether the column itself is gated open, since a prediction
         * that does not pan out should be weakened either way. */
        if (L->p.enable_decay) {
            u32 pn = pred_word & ~act_word;
            while (pn) {
                u32 cell = __builtin_ctz(pn);
                pn &= pn - 1u;
                for (u32 seg = 0; seg < segs; ++seg) {
                    u32 g = segment_global_index(L, col, cell, seg);
                    if (L->segment_meta[g].spike_count >= theta_d) {
                        push_update(L, g, UPDATE_DECAY);
                    }
                }
            }
        }
    }
}

static void learn_collect_topk(layer_t* L) {
    /* For TOPK there is no column-gating, no bursting. Active cells were
     * those with both ffwd and tau support. Reinforce their spiking
     * segments. Decay: predicted-but-inactive cells. */
    const u8 theta_d = L->p.htm.segment_spiking_threshold;
    const u8 segs = L->p.segments_per_cell;

    for (u32 col = 0; col < L->p.cols; ++col) {
        for (u32 cell = 0; cell < L->p.cells_per_col; ++cell) {
            u32 idx = encode_active_cell(L, col, cell);
            u32 is_active    = layer_bit_get(L->active, idx);
            u32 is_predicted = layer_bit_get(L->predicted, idx);
            if (is_active) {
                for (u32 seg = 0; seg < segs; ++seg) {
                    u32 g = segment_global_index(L, col, cell, seg);
                    if (L->segment_meta[g].spike_count >= theta_d) {
                        push_update(L, g, UPDATE_REINFORCE);
                    }
                }
            } else if (is_predicted && L->p.enable_decay) {
                for (u32 seg = 0; seg < segs; ++seg) {
                    u32 g = segment_global_index(L, col, cell, seg);
                    if (L->segment_meta[g].spike_count >= theta_d) {
                        push_update(L, g, UPDATE_DECAY);
                    }
                }
            }
        }
    }
}

void layer_learn(layer_t* L) {
    if (!L->p.enable_distal_learning) return;
    L->update_segments_count = 0;

    if (L->p.decision == DECIDE_BURST_OR_PREDICTED) {
        learn_collect_burst(L);
    } else {
        learn_collect_topk(L);
    }

    /* Apply Hebbian per (segment, op) pair via by-seg index. */
    const u8 inc = L->p.htm.perm_increment;
    const u8 dec = L->p.htm.perm_decrement;
    const u8 dcy = L->p.htm.perm_decay;

    for (u32 i = 0; i < L->update_segments_count; ++i) {
        u32 seg = L->update_segments[i];
        u8 op = L->update_segment_op[i];

        u32 a = L->by_seg_offset[seg];
        u32 b = L->by_seg_offset[seg + 1];

        for (u32 k = a; k < b; ++k) {
            connection_t* c = &L->connections[L->by_seg_data[k]];
            const input_stream_t* str = &L->streams[c->stream_id];
            /* Hebbian on prev-step activity of the source. */
            u32 src_active = (str->activity != NULL) &&
                             layer_bit_get(str->activity, c->source_index);

            if (op == UPDATE_REINFORCE) {
                c->permanence = src_active
                    ? safe_add_u8(c->permanence, inc)
                    : safe_sub_u8(c->permanence, dec);
            } else { /* UPDATE_DECAY */
                if (src_active && c->permanence >= str->perm_threshold) {
                    c->permanence = safe_sub_u8(c->permanence, dcy);
                }
            }
        }
    }
}

/* =====================================================================
 * Step orchestration + active_prev := active snapshot.
 *
 * Convention: outside of a step, `L->active` always holds the most
 * recent decision. Other layers' streams that point at `&L->active`
 * therefore read the latest activations regardless of when in the
 * step they run. `L->active_prev` is the snapshot of the previous
 * step's decision; `project` reads it (directly or via a self-pointing
 * stream).
 *
 * To preserve this invariant, we COPY active -> active_prev at the
 * start of the step, then write the new decision into active. No
 * pointer swapping. Cost: a small memcpy per step (active is 4 KiB
 * for 1024 cols).
 * ===================================================================== */

static void layer_snapshot_active(layer_t* L) {
    memcpy(L->active_prev, L->active, L->activity_words * sizeof(u32));
    if (L->active_cells_count > 0) {
        memcpy(L->active_prev_cells, L->active_cells,
               L->active_cells_count * sizeof(u32));
    }
    L->active_prev_cells_count = L->active_cells_count;
}

void layer_step(layer_t* L, const u32* active_columns) {
    layer_snapshot_active(L);
    layer_project(L);
    layer_predict(L);
    layer_decide(L, active_columns);
    layer_learn(L);
}

/* =====================================================================
 * L6 grid-cell shift (pre-projection hook).
 *
 * Permutes the layer's *previous* active state along a 2D movement
 * vector. After this call, layer_project will see the shifted
 * active_prev via any stream pointing at &L->active_prev[0].
 *
 * Mirrors the modulo-pow2 shift in legacy location_layer.c.
 * ===================================================================== */

void layer_shift_active_prev_grid(layer_t* L,
                                  vec2d movement,
                                  uvec2d log_scale,
                                  u32 log_cols_sqrt) {
    u32 cols_sqrt = 1u << log_cols_sqrt;
    u32 mask = cols_sqrt - 1u;

    /* Build new active_prev word-by-word into a scratch buffer. */
    u32* shifted = (u32*) calloc(L->activity_words, sizeof(u32));
    u32  new_count = 0;
    u32* new_list = (u32*) malloc(L->total_cells * sizeof(u32));

    for (u32 i = 0; i < L->active_prev_cells_count; ++i) {
        u32 idx = L->active_prev_cells[i];
        u32 col, cell;
        decode_active_cell(L, idx, &col, &cell);

        i32 x = (i32)(col & mask);
        i32 y = (i32)(col >> log_cols_sqrt);

        i32 new_x = (x + (movement.x >> log_scale.x)) & (i32) mask;
        i32 new_y = (y + (movement.y >> log_scale.y)) & (i32) mask;
        if (new_x < 0) new_x += cols_sqrt;
        if (new_y < 0) new_y += cols_sqrt;

        u32 new_col = (u32) new_x + ((u32) new_y << log_cols_sqrt);
        u32 new_idx = encode_active_cell(L, new_col, cell);

        if (L->p.cells_per_col == 1) {
            shifted[new_idx >> 5] |= (1u << (new_idx & 31u));
        } else {
            shifted[new_col] |= (1u << cell);
        }
        new_list[new_count++] = new_idx;
    }

    memcpy(L->active_prev, shifted, L->activity_words * sizeof(u32));
    free(shifted);

    /* Replace active_prev_cells with shifted list. */
    memcpy(L->active_prev_cells, new_list, new_count * sizeof(u32));
    L->active_prev_cells_count = new_count;
    free(new_list);
}

/* =====================================================================
 * Introspection
 * ===================================================================== */

u32 layer_connections_footprint_bytes(const layer_params_t* p, u8 num_streams) {
    (void) num_streams;
    u32 total_cells = (u32) p->cols * p->cells_per_col;
    u32 total_segments = total_cells * p->segments_per_cell;
    u32 n_distal = total_segments * p->conns_per_segment;
    u32 n_ffwd = total_cells * p->ffwd_conns_per_cell;
    u32 dist_arena = n_distal * (u32) sizeof(connection_t);
    u32 dist_idx   = n_distal * 2u * (u32) sizeof(u32)
                   + (total_segments + 1u) * (u32) sizeof(u32);
    u32 ffwd_arena = n_ffwd * (u32) sizeof(connection_t);
    u32 ffwd_idx   = n_ffwd * 2u * (u32) sizeof(u32)
                   + (total_cells + 1u) * (u32) sizeof(u32);
    return dist_arena + dist_idx + ffwd_arena + ffwd_idx;
}

u32 layer_state_footprint_bytes(const layer_params_t* p) {
    u32 total_cells = (u32) p->cols * p->cells_per_col;
    u32 total_segments = total_cells * p->segments_per_cell;
    u32 activity_words = (p->cells_per_col == 1)
                            ? ((p->cols + 31u) >> 5) : p->cols;
    u32 act_bytes = 3u * activity_words * (u32) sizeof(u32);
    u32 seg_acc   = total_segments * (u32) sizeof(u8);
    u32 seg_meta  = total_segments * (u32) sizeof(segment_meta_t);
    u32 cell_scratch = 2u * total_cells * (u32) sizeof(u8);
    u32 active_lists = 2u * total_cells * (u32) sizeof(u32);
    u32 update_bufs  = total_segments * ((u32) sizeof(u32) + (u32) sizeof(u8));
    return act_bytes + seg_acc + seg_meta + cell_scratch
         + active_lists + update_bufs;
}

void layer_print_memory_footprint(const layer_t* L) {
    u32 conns = layer_connections_footprint_bytes(&L->p, L->num_streams);
    u32 state = layer_state_footprint_bytes(&L->p);
    u32 total = conns + state;
    printf("-- layer (cols=%u, cells=%u, segs=%u, conns/seg=%u, ffwd/cell=%u):\n",
           L->p.cols, L->p.cells_per_col, L->p.segments_per_cell,
           L->p.conns_per_segment, L->p.ffwd_conns_per_cell);
    printf("   distal arena       %u MiB (%u KiB)\n",
           (L->n_distal * (u32) sizeof(connection_t)) >> 20,
           (L->n_distal * (u32) sizeof(connection_t)) >> 10);
    printf("   distal indices     %u MiB (%u KiB)\n",
           (L->n_distal * 2u * (u32) sizeof(u32)) >> 20,
           (L->n_distal * 2u * (u32) sizeof(u32)) >> 10);
    printf("   ffwd  arena+idx    %u KiB\n",
           (L->n_ffwd * ((u32) sizeof(connection_t) + 2u * (u32) sizeof(u32))) >> 10);
    printf("   state              %u KiB\n", state >> 10);
    printf("   total              %u MiB (%u KiB)\n", total >> 20, total >> 10);
}
