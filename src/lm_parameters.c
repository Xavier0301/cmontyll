#include "lm_parameters.h"

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
