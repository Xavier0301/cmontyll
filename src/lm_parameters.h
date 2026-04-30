#ifndef LM_PARAMETERS_H
#define LM_PARAMETERS_H

#include "types.h"
#include <stdio.h>

/* ============== HTM ============== */

typedef struct htm_params_t_ {
    u8 permanence_threshold;

    u8 segment_spiking_threshold;
    u8 predicted_threshold;

    u8 perm_increment;
    u8 perm_decrement;
    u8 perm_decay; // how much to decay predicted but inactive cells 
    // TODO: maybe add a decay period?
} htm_params_t;

void htm_print_params(htm_params_t p);

typedef struct extended_htm_params_t_ {
    u8 feedforward_permanence_threshold;
    u8 context_permanence_threshold;

    u8 feedforward_activation_threshold;
    u8 context_activation_threshold;

    u8 min_active_cells;
} extended_htm_params_t;

void htm_print_extended_params(extended_htm_params_t p);

/* ============== FEATURE LAYER ============== */

typedef struct feature_layer_params_t_ {
    u16 cols;
    u16 cells; // per columns

    u8 feature_segments; // per cell
    u8 location_segments; // per cell

    htm_params_t htm;
} feature_layer_params_t;

void feature_layer_print_params(feature_layer_params_t p);

/* ============== LOCATION LAYER ============== */

typedef struct location_layer_params_t_ {
    u32 cols; // cols represent high level grid cells, we have cols_sqrt x cols_sqrt cols in the network
    u32 log_cols_sqrt; /* cols represent high level grid cells, we have cols_sqrt x cols_sqrt cols in the network
        Having the sqrt columns in log form allows us to transform a column index i into the column location (x, y) */
    u32 cells; // cells per column, these represent unique 

    u8 location_segments;
    u8 feature_segments;

    htm_params_t htm;

    // the following determined how the movement vector is modified
    uvec2d log_scale;
} location_layer_params_t;

void location_layer_print_params(location_layer_params_t p);

/* ============== OUTPUT LAYER ============== */

typedef struct output_layer_params_t_ {
    u16 cells;
    u8 log_cells;

    u8 internal_context_segments;
    u8 external_context_segments;

    u16 external_cells;
    u8 log_external_cells;
    u8 external_lms;

    htm_params_t htm;
    extended_htm_params_t extended_htm;
} output_layer_params_t;

void output_layer_print_params(output_layer_params_t p);

/* ============== UNIFIED LAYER CONSTRUCTORS ==============
 *
 * Each constructor produces a `layer_params_t` (defined in layer.h)
 * configured for one of the four cortical-column cell-style layers.
 * Stream wiring (which other layers feed in) is the caller's job and
 * happens at `layer_init_connections` time -- the params returned here
 * only describe shape and decision rule.
 *
 * The constructors are defined in lm_parameters.c. Header forward-decls
 * `struct layer_params_t_` so callers that only need the legacy params
 * can include this header without dragging layer.h in. */
struct layer_params_t_;

struct layer_params_t_ make_l4_params(u16 cols, u8 cells_per_col,
                                      u8 feature_segments, u8 location_segments,
                                      u16 conns_per_segment,
                                      htm_params_t htm);

struct layer_params_t_ make_l6_params(u16 cols, u8 cells_per_col,
                                      u8 location_segments, u8 feature_segments,
                                      u16 conns_per_segment,
                                      htm_params_t htm);

struct layer_params_t_ make_l3_params(u16 cells,
                                      u8 internal_context_segments,
                                      u8 external_context_segments,
                                      u16 conns_per_segment,
                                      u16 ffwd_conns_per_cell,
                                      u16 top_k,
                                      htm_params_t htm,
                                      extended_htm_params_t ext);

#endif
