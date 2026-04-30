/* =====================================================================
 * tests/test_layer.c -- per-phase unit tests for the unified layer_t.
 *
 * Each test sets up a minimal layer, drives it through one or two
 * phases, and asserts on the layer's internal state. The tests double
 * as canonical usage examples for callers integrating layer_t into a
 * new system (see docs/integration_guide.md).
 *
 * Build & run:
 *     make test
 *     ./test
 *
 * Strategy: rather than try to control RNG draws (brittle), we let
 * `layer_init_connections` draw randomly, then read what it drew off
 * `L.connections[i]` and drive the test from there.
 * ===================================================================== */

#include <stdio.h>
#include <string.h>

#include "layer.h"

/* ============== test runner ============== */

static int n_pass = 0, n_fail = 0;
#define CHECK(cond, msg) do {                                           \
    if (cond) {                                                         \
        ++n_pass;                                                       \
    } else {                                                            \
        ++n_fail;                                                       \
        fprintf(stderr, "FAIL %s:%d  %s\n", __FILE__, __LINE__, msg);   \
    }                                                                   \
} while (0)

#define RUN(test) do { \
    fprintf(stderr, "running %s ...\n", #test);                         \
    int before = n_fail;                                                \
    test();                                                             \
    fprintf(stderr, "  %s\n", (n_fail == before) ? "ok" : "FAILED");    \
} while (0)

/* ============== fixtures ============== */

static htm_params_t default_htm(void) {
    return (htm_params_t) {
        .permanence_threshold = 128,       /* mid-range u8 */
        .segment_spiking_threshold = 1,    /* theta_d = 1 */
        .predicted_threshold = 1,
        .perm_increment = 10,
        .perm_decrement = 4,
        .perm_decay = 1,
    };
}

/* Tiny BURST layer: 4 cols * 2 cells * 1 segment * 1 conn = 8 connections.
 * Self-recurrent (single distal stream pointing at L.active_prev). */
static void make_tiny_burst(layer_t* L, u32* seed) {
    layer_params_t p = (layer_params_t) {
        .cols = 4,
        .cells_per_col = 2,
        .segments_per_cell = 1,
        .conns_per_segment = 1,
        .ffwd_conns_per_cell = 0,
        .decision = DECIDE_BURST_OR_PREDICTED,
        .top_k = 0,
        .htm = default_htm(),
        .enable_distal_learning = 1,
        .enable_decay = 1,
        .enable_ffwd_learning = 0,
    };
    layer_init_state(L, p);
    input_stream_t s = (input_stream_t) {
        .kind = STREAM_DISTAL,
        .activity = L->active_prev,
        .source_cols = p.cols,
        .source_cells_per_col = p.cells_per_col,
        .source_cells = 0,
        .segments_assigned = 1,
        .perm_threshold = p.htm.permanence_threshold,
        .activation_threshold = 0,
        .name = "self",
    };
    layer_init_connections(L, &s, 1, seed);
}

/* TOPK (flat) layer: 8 cells, 2 distal segments per cell, k = 2.
 * One proximal stream from a 4-col 2-cell source, one distal
 * self-recurrent (flat). */
static void make_tiny_topk(layer_t* L, u32* feat_active_buf, u32* seed) {
    layer_params_t p = (layer_params_t) {
        .cols = 8,                        /* flat: cells_per_col == 1 */
        .cells_per_col = 1,
        /* segments_per_cell bounds the histogram in find_kth_largest_u8;
         * pick a value large enough to keep test tau scores in range. */
        .segments_per_cell = 4,
        .conns_per_segment = 1,
        .ffwd_conns_per_cell = 1,
        .decision = DECIDE_TOPK_PREDICTED_FFWD,
        .top_k = 2,
        .htm = default_htm(),
        .enable_distal_learning = 1,
        .enable_decay = 0,
        .enable_ffwd_learning = 0,
    };
    layer_init_state(L, p);
    /* feat_active_buf = caller-owned 4-word per-col bitarray (2 cells/col x 4 cols). */
    memset(feat_active_buf, 0, 4 * sizeof(u32));

    input_stream_t streams[2];
    streams[0] = (input_stream_t) {
        .kind = STREAM_PROXIMAL,
        .activity = feat_active_buf,
        .source_cols = 4,
        .source_cells_per_col = 2,
        .source_cells = 0,
        .segments_assigned = 0,
        .perm_threshold = p.htm.permanence_threshold,
        .activation_threshold = 0,                /* theta_F = 0 for the test */
        .name = "ffwd",
    };
    streams[1] = (input_stream_t) {
        .kind = STREAM_DISTAL,
        .activity = L->active_prev,
        .source_cols = p.cols,                    /* flat */
        .source_cells_per_col = 1,
        .source_cells = 0,
        .segments_assigned = 4,                   /* must == segments_per_cell */
        .perm_threshold = p.htm.permanence_threshold,
        .activation_threshold = 0,
        .name = "self",
    };
    layer_init_connections(L, streams, 2, seed);
}

/* ============== tests ============== */

/* layer_project: a single connection contributes 1 to its target
 * accumulator iff the source bit is set AND permanence >= threshold.
 * The active-prev sparse list drives iteration. */
static void test_project(void) {
    u32 seed = 42;
    layer_t L; make_tiny_burst(&L, &seed);

    /* Pick connection 0; observe what (source, segment) it drew. */
    u32 src_idx = L.connections[0].source_index;
    u32 seg     = L.connections[0].segment_index;
    L.connections[0].permanence = 200;          /* well above threshold */

    /* Mark the source cell active in the self-recurrent stream's
     * activity bitarray AND in the sparse active_prev_cells list. */
    L.active_prev[src_idx >> 5] |= (1u << (src_idx & 31u));
    L.active_prev_cells[0] = src_idx;
    L.active_prev_cells_count = 1;

    layer_project(&L);
    CHECK(L.segment_accumulators.data[seg] >= 1,
          "project: connection above threshold contributes to accumulator");

    /* Drop permanence; project again; expect contribution to vanish. */
    L.connections[0].permanence = 10;
    layer_project(&L);
    /* Other connections targeting `seg` may still contribute; but with
     * conns_per_segment=1, only connection 0 targets it. */
    CHECK(L.segment_accumulators.data[seg] == 0,
          "project: below-threshold connection does not contribute");

    layer_free(&L);
}

/* layer_predict: stuff segment_accumulators directly, run predict,
 * check spike_count saved, segment_spikes computed, predicted set. */
static void test_predict(void) {
    u32 seed = 1;
    layer_t L; make_tiny_burst(&L, &seed);

    /* Cell 0 of col 0 has segment_global_index 0 (cells_per_col=2,
     * segments_per_cell=1 -> seg(c, 0, 0) = c * 2 + 0). */
    L.segment_accumulators.data[0] = 5;     /* col 0, cell 0, seg 0 */
    L.segment_accumulators.data[1] = 0;     /* col 0, cell 1, seg 0 */

    layer_predict(&L);

    CHECK(L.segment_meta[0].spike_count == 5, "predict: spike_count saved");
    CHECK(L.segment_spikes[0] == 1, "predict: tau >= theta_d for cell (0,0)");
    CHECK(L.segment_spikes[1] == 0, "predict: tau == 0 for cell (0,1)");
    CHECK((L.predicted[0] & 0x1u) == 0x1u, "predict: cell (0,0) predicted bit set");
    CHECK((L.predicted[0] & 0x2u) == 0x0u, "predict: cell (0,1) predicted bit clear");

    layer_free(&L);
}

/* layer_decide BURST: gated by active_columns. With no predictions,
 * every cell of an active column bursts. With one predicted cell, only
 * that cell fires. */
static void test_decide_burst(void) {
    u32 seed = 5;
    layer_t L; make_tiny_burst(&L, &seed);

    /* active_columns: packed-32 bitarray over cols (1 word for 4 cols). */
    u32 active_cols[1] = {0};
    active_cols[0] = (1u << 1);             /* col 1 active */

    /* No predictions -> col 1 bursts: cells 0 and 1 both active. */
    memset(L.predicted, 0, L.activity_words * sizeof(u32));
    layer_decide(&L, active_cols);
    CHECK(L.active[1] == 0x3u, "burst: both cells active when no prediction");
    CHECK(L.active_cells_count == 2, "burst: 2 active cells");
    CHECK(L.active[0] == 0 && L.active[2] == 0 && L.active[3] == 0,
          "burst: gated columns stay silent");

    /* With prediction on col 1 cell 0: only cell 0 fires. */
    memset(L.predicted, 0, L.activity_words * sizeof(u32));
    L.predicted[1] = 0x1u;
    layer_decide(&L, active_cols);
    CHECK(L.active[1] == 0x1u, "burst+predicted: only predicted cell active");
    CHECK(L.active_cells_count == 1, "burst+predicted: 1 active cell");

    layer_free(&L);
}

/* layer_decide TOPK: tau scores drive selection, ffwd_overlap gates.
 * With 8 cells, k=2, three cells with non-zero tau, the two highest tau
 * win iff their ffwd_overlap >= theta_F. */
static void test_decide_topk(void) {
    u32 seed = 3;
    u32 feat_active[4];
    layer_t L; make_tiny_topk(&L, feat_active, &seed);

    /* Stuff segment_spikes directly (bypass predict). Values must be
     * within [0, segments_per_cell] = [0, 4] for find_kth_largest_u8's
     * histogram to see them. */
    memset(L.segment_spikes, 0, L.total_cells);
    L.segment_spikes[0] = 2;    /* candidate (3rd largest, loses) */
    L.segment_spikes[3] = 4;    /* top */
    L.segment_spikes[5] = 3;    /* 2nd largest, wins */

    /* compute_ffwd_overlap_sparse runs inside layer_decide and reads
     * the proximal stream. Drive it via feat_active.
     * Need cells with ffwd >= theta_F (theta_F = 0 here, so any >= 0
     * passes -- effectively every cell qualifies). */
    /* Make all proximal sources inactive so ffwd_overlap is 0 for all
     * cells. With theta_F = 0, the condition `fov >= theta_F` is still
     * true; tau alone decides. */
    memset(feat_active, 0, sizeof(feat_active));

    layer_decide(&L, NULL);
    /* Top 2 by tau: cells 3 (9) and 5 (7). Cell 0 (5) loses. */
    CHECK(layer_bit_get(L.active, 3) == 1, "topk: highest tau wins");
    CHECK(layer_bit_get(L.active, 5) == 1, "topk: 2nd-highest tau wins");
    CHECK(layer_bit_get(L.active, 0) == 0, "topk: 3rd-highest tau loses");

    layer_free(&L);
}

/* layer_learn REINFORCE on a correctly-predicted cell.
 *
 * Setup a connection so its post cell ends up active && predicted with
 * a spiking segment, and its source bit is set in the stream's
 * activity. Permanence should jump by perm_increment. */
static void test_learn_reinforce_correct(void) {
    u32 seed = 9;
    layer_t L; make_tiny_burst(&L, &seed);

    /* Pick connection 0; learn what it points at. */
    u32 src_idx = L.connections[0].source_index;
    u32 seg     = L.connections[0].segment_index;
    /* segments_per_cell = 1, so seg == cell global index. */
    u32 cell_g = seg;
    u32 col = cell_g / L.p.cells_per_col;
    u32 cell_in_col = cell_g % L.p.cells_per_col;

    L.connections[0].permanence = 100;

    /* Mark cell active && predicted, and the segment as spiking. */
    memset(L.active, 0, L.activity_words * sizeof(u32));
    memset(L.predicted, 0, L.activity_words * sizeof(u32));
    L.active[col]    = 1u << cell_in_col;
    L.predicted[col] = 1u << cell_in_col;
    L.segment_meta[seg].spike_count = 5;        /* >= theta_d=1 */

    /* Mark the source bit in the self-recurrent stream's activity
     * (which IS L.active_prev). */
    L.active_prev[src_idx >> 5] |= (1u << (src_idx & 31u));

    layer_learn(&L);

    CHECK(L.connections[0].permanence == 110,
          "REINFORCE: +perm_increment on active source incident");

    /* Now repeat with source INactive: should -decrement instead. */
    L.connections[0].permanence = 100;
    memset(L.active_prev, 0, L.activity_words * sizeof(u32));
    layer_learn(&L);
    CHECK(L.connections[0].permanence == 96,
          "REINFORCE: -perm_decrement on inactive source incident");

    layer_free(&L);
}

/* layer_learn DECAY on predicted-but-inactive cells. Source bit must be
 * set AND permanence must be >= stream.perm_threshold for decay to
 * apply. */
static void test_learn_decay(void) {
    u32 seed = 11;
    layer_t L; make_tiny_burst(&L, &seed);

    u32 src_idx = L.connections[0].source_index;
    u32 seg     = L.connections[0].segment_index;
    u32 cell_g = seg;
    u32 col = cell_g / L.p.cells_per_col;
    u32 cell_in_col = cell_g % L.p.cells_per_col;

    L.connections[0].permanence = 200;          /* above stream threshold */

    /* predicted && !active && segment spiking */
    memset(L.active, 0, L.activity_words * sizeof(u32));
    memset(L.predicted, 0, L.activity_words * sizeof(u32));
    L.predicted[col] = 1u << cell_in_col;
    L.segment_meta[seg].spike_count = 5;

    /* Source bit active. */
    L.active_prev[src_idx >> 5] |= (1u << (src_idx & 31u));

    layer_learn(&L);
    CHECK(L.connections[0].permanence == 199,
          "DECAY: -perm_decay on active source incident above threshold");

    /* Below threshold: no decay. */
    L.connections[0].permanence = 50;       /* < perm_threshold=128 */
    layer_learn(&L);
    CHECK(L.connections[0].permanence == 50,
          "DECAY: skipped when permanence below threshold");

    layer_free(&L);
}

/* Saturation at u8 bounds. */
static void test_learn_saturation(void) {
    u32 seed = 13;
    layer_t L; make_tiny_burst(&L, &seed);

    u32 src_idx = L.connections[0].source_index;
    u32 seg     = L.connections[0].segment_index;
    u32 col = seg / L.p.cells_per_col;
    u32 cell_in_col = seg % L.p.cells_per_col;

    /* Saturate up. */
    L.connections[0].permanence = 250;
    L.active[col]    = 1u << cell_in_col;
    L.predicted[col] = 1u << cell_in_col;
    L.segment_meta[seg].spike_count = 5;
    L.active_prev[src_idx >> 5] |= (1u << (src_idx & 31u));
    layer_learn(&L);
    CHECK(L.connections[0].permanence == 255, "REINFORCE: clamps at 255");

    /* Saturate down. */
    L.connections[0].permanence = 2;
    memset(L.active_prev, 0, L.activity_words * sizeof(u32));   /* source inactive */
    layer_learn(&L);
    CHECK(L.connections[0].permanence == 0, "REINFORCE: clamps at 0");

    layer_free(&L);
}

/* layer_shift_active_prev_grid: toroidal pow-2 shift on a per-column
 * layer's active_prev. With 4 cols (cols_sqrt = 2, log_cols_sqrt = 1)
 * and movement = (1, 0), col 0 -> col 1, col 1 -> col 0 (wrap),
 * col 2 -> col 3, col 3 -> col 2. */
static void test_grid_shift(void) {
    u32 seed = 17;
    layer_t L; make_tiny_burst(&L, &seed);

    /* Set col 0 cell 0 active in active_prev. */
    memset(L.active_prev, 0, L.activity_words * sizeof(u32));
    L.active_prev[0] = 0x1u;
    L.active_prev_cells[0] = 0;
    L.active_prev_cells_count = 1;

    vec2d move = { .x = 1, .y = 0 };
    uvec2d log_scale = { .x = 0, .y = 0 };          /* unscaled movement */
    u32 log_cols_sqrt = 1;                          /* 2x2 grid */

    layer_shift_active_prev_grid(&L, move, log_scale, log_cols_sqrt);

    /* col 0 (x=0,y=0) shifted by (+1,0) -> (x=1,y=0) -> col 1. */
    CHECK(L.active_prev[1] == 0x1u, "grid shift: col 0 -> col 1");
    CHECK(L.active_prev[0] == 0,    "grid shift: col 0 cleared");

    layer_free(&L);
}

/* ============== entry ============== */

int main(void) {
    fprintf(stderr, "=== layer_t unit tests ===\n");

    RUN(test_project);
    RUN(test_predict);
    RUN(test_decide_burst);
    RUN(test_decide_topk);
    RUN(test_learn_reinforce_correct);
    RUN(test_learn_decay);
    RUN(test_learn_saturation);
    RUN(test_grid_shift);

    fprintf(stderr, "\n=== %d passed, %d failed ===\n", n_pass, n_fail);
    return n_fail == 0 ? 0 : 1;
}
