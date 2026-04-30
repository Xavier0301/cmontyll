#include "lm_parameters.h"
#include "layer.h"

/* ============== HTM ============== */

void htm_print_params(htm_params_t p) {
#if PRINT == 2
    printf("htm_params_t:\n");
    printf("\tpermanence_threshold            = %u (%.3f)\n", p.permanence_threshold, p.permanence_threshold / 255.0f);
    printf("\tsegment_spiking_threshold       = %u\n", p.segment_spiking_threshold);
    printf("\tperm_increment                  = %u (%.4f)\n", p.perm_increment, p.perm_increment / 255.0f);
    printf("\tperm_decrement                  = %u (%.4f)\n", p.perm_decrement, p.perm_decrement / 255.0f);
    printf("\tperm_decay                      = %u (%.4f)\n", p.perm_decay, p.perm_decay / 255.0f);
#else
    (void) p;
#endif
}

void htm_print_extended_params(extended_htm_params_t p) {
#if PRINT == 2
    printf("extended_htm_params_t:\n");
    printf("\tfeedforward_permanence_threshold = %u (%.3f)\n", p.feedforward_permanence_threshold, p.feedforward_permanence_threshold / 255.0f);
    printf("\tcontext_permanence_threshold     = %u (%.3f)\n", p.context_permanence_threshold, p.context_permanence_threshold / 255.0f);
    printf("\tfeedforward_activation_threshold = %u\n", p.feedforward_activation_threshold);
    printf("\tcontext_activation_threshold     = %u\n", p.context_activation_threshold);
    printf("\tmin_active_cells                 = %u\n", p.min_active_cells);
#else
    (void) p;
#endif
}

/* ============== FEATURE LAYER ============== */

void feature_layer_print_params(feature_layer_params_t p) {
#if PRINT == 2
    printf("feature_layer_params_t:\n");
    printf("\tcols                           = %u\n", p.cols);
    printf("\tcells                          = %u\n", p.cells);
    printf("\tfeature_segments               = %u\n", p.feature_segments);
    printf("\tlocation_segments              = %u\n", p.location_segments);
    htm_print_params(p.htm);
#else
    (void) p;
#endif
}

/* ============== LOCATION LAYER ============== */

void location_layer_print_params(location_layer_params_t p) {
#if PRINT == 2
    printf("location_layer_params_t:\n");
    printf("\tcols                           = %u\n", p.cols);
    printf("\tlog_cols_sqrt                  = %u\n", p.log_cols_sqrt);
    printf("\tcells                          = %u\n", p.cells);
    printf("\tlocation_segments              = %u\n", p.location_segments);
    printf("\tfeature_segments               = %u\n", p.feature_segments);
    printf("\tlog_scale                      = { x=%u, y=%u }\n", p.log_scale.x, p.log_scale.y);
    htm_print_params(p.htm);
#else
    (void) p;
#endif
}

/* ============== OUTPUT LAYER ============== */

void output_layer_print_params(output_layer_params_t p) {
#if PRINT == 2
    printf("output_layer_params_t:\n");
    printf("\tcells                           = %u\n", p.cells);
    printf("\tinternal_context_segments       = %u\n", p.internal_context_segments);
    printf("\texternal_context_segments       = %u\n", p.external_context_segments);
    // reuse HTM printers
    htm_print_params(p.htm);
    htm_print_extended_params(p.extended_htm);
#else
    (void) p;
#endif
}

/* ============== UNIFIED LAYER CONSTRUCTORS ==============
 *
 * The htm_params_t embedded in each `layer_params_t` carries the
 * permanence and threshold knobs that survive to the unified layer's
 * project / predict / learn rules. Per-stream overrides
 * (perm_threshold, activation_threshold) live on the input_stream_t
 * descriptors built by the caller. */

struct layer_params_t_ make_l4_params(u16 cols, u8 cells_per_col,
                                      u8 feature_segments, u8 location_segments,
                                      u16 conns_per_segment,
                                      htm_params_t htm) {
    layer_params_t p = (layer_params_t) {
        .cols = cols,
        .cells_per_col = cells_per_col,
        .segments_per_cell = (u8)(feature_segments + location_segments),
        .conns_per_segment = conns_per_segment,
        .ffwd_conns_per_cell = 0,        /* L4 proximal is the column gate, not a learnable arena */
        .decision = DECIDE_BURST_OR_PREDICTED,
        .top_k = 0,
        .htm = htm,
        .enable_distal_learning = 1,
        .enable_decay = 1,
        .enable_ffwd_learning = 0,
    };
    return p;
}

struct layer_params_t_ make_l6_params(u16 cols, u8 cells_per_col,
                                      u8 location_segments, u8 feature_segments,
                                      u16 conns_per_segment,
                                      htm_params_t htm) {
    layer_params_t p = (layer_params_t) {
        .cols = cols,
        .cells_per_col = cells_per_col,
        .segments_per_cell = (u8)(location_segments + feature_segments),
        .conns_per_segment = conns_per_segment,
        .ffwd_conns_per_cell = 0,
        .decision = DECIDE_BURST_OR_PREDICTED,
        .top_k = 0,
        .htm = htm,
        .enable_distal_learning = 1,
        .enable_decay = 1,
        .enable_ffwd_learning = 0,
    };
    return p;
}

struct layer_params_t_ make_l3_params(u16 cells,
                                      u8 internal_context_segments,
                                      u8 external_context_segments,
                                      u16 conns_per_segment,
                                      u16 ffwd_conns_per_cell,
                                      u16 top_k,
                                      htm_params_t htm,
                                      extended_htm_params_t ext) {
    (void) ext;  /* extended thresholds applied at stream-config time */
    layer_params_t p = (layer_params_t) {
        .cols = cells,
        .cells_per_col = 1,             /* L3 is a flat layer */
        .segments_per_cell = (u8)(internal_context_segments + external_context_segments),
        .conns_per_segment = conns_per_segment,
        .ffwd_conns_per_cell = ffwd_conns_per_cell,
        .decision = DECIDE_TOPK_PREDICTED_FFWD,
        .top_k = top_k,
        .htm = htm,
        .enable_distal_learning = 1,
        .enable_decay = 0,              /* legacy L3 has no decay */
        .enable_ffwd_learning = 0,      /* legacy TODO */
    };
    return p;
}
