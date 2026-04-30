/* =====================================================================
 * Learning module: thin orchestrator over three unified `layer_t` instances.
 *
 * Wiring (v1):
 *   L4 (feature):
 *     distal s0 -- self-recurrent (reads L4.active_prev, feat segs)
 *     distal s1 -- L6 location    (reads L6.active,      loc  segs)
 *     gate -- packed-32 mirror of features.active_columns (l4_col_gate)
 *
 *   L6 (location, BURST_OR_PREDICTED):
 *     distal s0 -- self-recurrent (reads L6.active_prev, loc  segs)
 *     distal s1 -- L4 feature     (reads L4.active,      feat segs)
 *     pre-step -- grid shift on active_prev by movement vector
 *     gate -- post-shift cols with any active cell (l6_col_gate)
 *
 *   L3 (output, TOPK_PREDICTED_FFWD, flat cells_per_col == 1):
 *     proximal  -- ffwd from L4.active                     (overlap theta_F)
 *     distal s0 -- self-recurrent internal context (reads L3.active_prev)
 *     external context segments are placeholders in v1 (split.l3_external_segments == 0).
 *
 * Per-step ordering preserves the convention that `&layer.active`
 * always exposes the latest decision:
 *   1. Pack pooler output -> l4_col_gate.
 *   2. L6: snapshot, grid-shift active_prev, derive l6_col_gate from
 *      shifted active_prev, project, predict, decide(l6_col_gate), learn.
 *      (Manual phases because the shift sits between snapshot and project.)
 *   3. L4: layer_step(l4_col_gate). Reads L6.active (just updated) and
 *      its own active_prev (snapshotted internally).
 *   4. L3: layer_step(NULL). Reads L4.active (just updated).
 * ===================================================================== */

#include "learning_module.h"

#include <string.h>
#include <assert.h>

/* ============== private helpers ============== */

static u32 lm_col_gate_words(u32 cols) {
    return (cols + 31u) >> 5;
}

/* Pack a u8-per-col sparse boolean activity vector into a packed-32
 * column gate. */
static void pack_col_gate_from_u8(u32* gate, u32 gate_words,
                                  const u8* active_columns, u32 num_cols) {
    memset(gate, 0, gate_words * sizeof(u32));
    for (u32 c = 0; c < num_cols; ++c) {
        if (active_columns[c]) {
            gate[c >> 5] |= (1u << (c & 31u));
        }
    }
}

/* Derive a packed-32 column gate from a per-column packed-cells activity
 * bitarray (one u32 word per column, cells laid out as bits). A column
 * is gated open iff at least one of its `cells_per_col` cells is set. */
static void derive_col_gate_from_per_col_activity(u32* gate, u32 gate_words,
                                                  const u32* per_col_activity,
                                                  u32 cols, u8 cells_per_col) {
    memset(gate, 0, gate_words * sizeof(u32));
    u32 cell_mask = (cells_per_col >= 32) ? 0xFFFFFFFFu
                                          : ((1u << cells_per_col) - 1u);
    for (u32 col = 0; col < cols; ++col) {
        if (per_col_activity[col] & cell_mask) {
            gate[col >> 5] |= (1u << (col & 31u));
        }
    }
}

/* ============== init / free ============== */

void init_learning_module(
    learning_module* lm,
    layer_params_t l4_p,
    layer_params_t l6_p,
    layer_params_t l3_p,
    lm_segment_split_t split,
    u32 pooler_cols,
    uvec2d l6_log_scale,
    u32 l6_log_cols_sqrt,
    u32* seed
) {
    /* v1 invariants: pooler width matches L4/L6 column count, and the
     * passed splits add up to the layer's segments_per_cell. */
    assert(l4_p.cols == pooler_cols);
    assert(l6_p.cols == pooler_cols);
    assert((u8)(split.l4_feature_segments + split.l4_location_segments)
           == l4_p.segments_per_cell);
    assert((u8)(split.l6_location_segments + split.l6_feature_segments)
           == l6_p.segments_per_cell);
    assert((u8)(split.l3_internal_segments + split.l3_external_segments)
           == l3_p.segments_per_cell);
    /* External context not wired in v1. */
    assert(split.l3_external_segments == 0);

    lm->l6_log_scale = l6_log_scale;
    lm->l6_log_cols_sqrt = l6_log_cols_sqrt;

    /* Phase 1: allocate state for all three layers. After this the
     * activity bitarrays exist and can be pointed at by other layers'
     * streams (including self-recurrent ones). */
    layer_init_state(&lm->l4, l4_p);
    layer_init_state(&lm->l6, l6_p);
    layer_init_state(&lm->l3, l3_p);

    /* Phase 2: column gates. */
    lm->col_gate_words = lm_col_gate_words(pooler_cols);
    lm->l4_col_gate = (u32*) calloc(lm->col_gate_words, sizeof(u32));
    lm->l6_col_gate = (u32*) calloc(lm->col_gate_words, sizeof(u32));

    /* Phase 3: build streams now that all three layers exist. */
    u8 perm_threshold_l4 = l4_p.htm.permanence_threshold;
    u8 perm_threshold_l6 = l6_p.htm.permanence_threshold;
    u8 perm_threshold_l3 = l3_p.htm.permanence_threshold;

    /* ---- L6 streams ---- */
    {
        input_stream_t l6_streams[2];
        l6_streams[0] = (input_stream_t) {
            .kind = STREAM_DISTAL,
            .activity = lm->l6.active_prev,
            .source_cols = l6_p.cols,
            .source_cells_per_col = l6_p.cells_per_col,
            .source_cells = 0,
            .segments_assigned = split.l6_location_segments,
            .perm_threshold = perm_threshold_l6,
            .activation_threshold = 0,
            .name = "l6.self",
        };
        l6_streams[1] = (input_stream_t) {
            .kind = STREAM_DISTAL,
            .activity = lm->l4.active,
            .source_cols = l4_p.cols,
            .source_cells_per_col = l4_p.cells_per_col,
            .source_cells = 0,
            .segments_assigned = split.l6_feature_segments,
            .perm_threshold = perm_threshold_l6,
            .activation_threshold = 0,
            .name = "l6.from_l4",
        };
        layer_init_connections(&lm->l6, l6_streams, 2, seed);
    }

    /* ---- L4 streams ---- */
    {
        input_stream_t l4_streams[2];
        l4_streams[0] = (input_stream_t) {
            .kind = STREAM_DISTAL,
            .activity = lm->l4.active_prev,
            .source_cols = l4_p.cols,
            .source_cells_per_col = l4_p.cells_per_col,
            .source_cells = 0,
            .segments_assigned = split.l4_feature_segments,
            .perm_threshold = perm_threshold_l4,
            .activation_threshold = 0,
            .name = "l4.self",
        };
        l4_streams[1] = (input_stream_t) {
            .kind = STREAM_DISTAL,
            .activity = lm->l6.active,
            .source_cols = l6_p.cols,
            .source_cells_per_col = l6_p.cells_per_col,
            .source_cells = 0,
            .segments_assigned = split.l4_location_segments,
            .perm_threshold = perm_threshold_l4,
            .activation_threshold = 0,
            .name = "l4.from_l6",
        };
        layer_init_connections(&lm->l4, l4_streams, 2, seed);
    }

    /* ---- L3 streams ---- *
     * L3 is flat (cells_per_col == 1). One proximal stream from L4
     * supplies the ffwd overlap; one distal stream is self-recurrent
     * internal context. External-context distal is omitted in v1. */
    {
        u8 num = 0;
        input_stream_t l3_streams[2];
        l3_streams[num++] = (input_stream_t) {
            .kind = STREAM_PROXIMAL,
            .activity = lm->l4.active,
            .source_cols = l4_p.cols,
            .source_cells_per_col = l4_p.cells_per_col,
            .source_cells = 0,
            .segments_assigned = 0,
            .perm_threshold = perm_threshold_l3,
            /* theta_F lives on the proximal stream. The L3 ffwd
             * activation threshold from extended_htm_params_t is
             * passed via lm_parameters and honored here at decide. */
            .activation_threshold = 0,
            .name = "l3.ffwd_from_l4",
        };
        if (split.l3_internal_segments > 0) {
            l3_streams[num++] = (input_stream_t) {
                .kind = STREAM_DISTAL,
                .activity = lm->l3.active_prev,
                .source_cols = l3_p.cols,    /* flat: source_cols = total cells */
                .source_cells_per_col = 1,
                .source_cells = 0,
                .segments_assigned = split.l3_internal_segments,
                .perm_threshold = perm_threshold_l3,
                .activation_threshold = 0,
                .name = "l3.self",
            };
        }
        layer_init_connections(&lm->l3, l3_streams, num, seed);
    }
}

void free_learning_module(learning_module* lm) {
    layer_free(&lm->l4);
    layer_free(&lm->l6);
    layer_free(&lm->l3);
    free(lm->l4_col_gate);
    free(lm->l6_col_gate);
    lm->l4_col_gate = NULL;
    lm->l6_col_gate = NULL;
}

/* ============== step ============== */

void learning_module_step(
    learning_module* lm,
    features_t features,
    vec2d movement,
    lmat_u32* external_output_layer_activations
) {
    (void) external_output_layer_activations;  /* v1: external context not wired */

    /* 1. Pack pooler output into the L4 column gate. */
    pack_col_gate_from_u8(lm->l4_col_gate, lm->col_gate_words,
                          features.active_columns, features.num_columns);

    /* 2. L6: snapshot -> grid shift -> derive gate -> project / predict
     *    / decide / learn. Manual phasing because the shift must sit
     *    between snapshot and project. */
    {
        /* Snapshot active -> active_prev (preserves the convention
         * that &L->active always holds the latest decision). The
         * canonical helper is internal to layer.c, but a memcpy here
         * is equivalent. */
        memcpy(lm->l6.active_prev, lm->l6.active,
               lm->l6.activity_words * sizeof(u32));
        lm->l6.active_prev_cells_count = lm->l6.active_cells_count;
        if (lm->l6.active_cells_count > 0) {
            memcpy(lm->l6.active_prev_cells, lm->l6.active_cells,
                   lm->l6.active_cells_count * sizeof(u32));
        }

        /* Apply the toroidal pow-2 shift to active_prev. */
        layer_shift_active_prev_grid(&lm->l6, movement,
                                     lm->l6_log_scale,
                                     lm->l6_log_cols_sqrt);

        /* Derive the L6 column gate from the shifted active_prev:
         * a column is gated iff at least one of its cells was active
         * (post-shift). */
        derive_col_gate_from_per_col_activity(
            lm->l6_col_gate, lm->col_gate_words,
            lm->l6.active_prev, lm->l6.p.cols, lm->l6.p.cells_per_col);

        layer_project(&lm->l6);
        layer_predict(&lm->l6);
        layer_decide(&lm->l6, lm->l6_col_gate);
        layer_learn(&lm->l6);
    }

    /* 3. L4: standard step. Reads L6.active (just decided) for cross
     *    context and its own active_prev for self-recurrence. */
    layer_step(&lm->l4, lm->l4_col_gate);

    /* 4. L3: standard step. TOPK decision ignores active_columns. */
    layer_step(&lm->l3, NULL);
}
