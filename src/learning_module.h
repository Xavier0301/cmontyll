#ifndef LEARNING_MODULE_H
#define LEARNING_MODULE_H

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>

#include "math.h"

#include "types.h"
#include "location.h"
#include "interfaces.h"
#include "lmat.h"

#include "layer.h"
#include "lm_parameters.h"

/* =====================================================================
 * Learning module: a thousand-brains-style cortical column.
 *
 * Three unified `layer_t` instances stand in for L4 (feature), L6
 * (location, with grid-cell shift) and L3 (output / pose vote). The
 * pooler upstream is owned elsewhere (sensor module). The streams
 * connecting these layers are wired explicitly at init time and never
 * change after that -- per-step there is just an active-columns
 * repack and the L6 grid shift.
 * ===================================================================== */

typedef struct learning_module_ {
    layer_t l4;     /* feature        : BURST_OR_PREDICTED, gated by pooler */
    layer_t l6;     /* location       : BURST_OR_PREDICTED, gated by movement */
    layer_t l3;     /* output / vote  : TOPK_PREDICTED_FFWD */

    /* Packed-32 column gate for L4: bit i = pooler activated col i. */
    u32* l4_col_gate;
    /* Packed-32 column gate for L6: bit i = post-shift L6 col i has any active cell. */
    u32* l6_col_gate;
    u32  col_gate_words;        /* (cols + 31) >> 5 */

    /* Cached L6 grid-shift parameters. */
    uvec2d l6_log_scale;
    u32    l6_log_cols_sqrt;
} learning_module;

/* Per-layer distal-segment splits.
 *
 * Each layer's `segments_per_cell` is partitioned across its distal
 * input streams. The unified `layer_params_t` only carries the total;
 * the wiring needs the split, so callers pass it here. The split fields
 * must satisfy:  feat + loc == l4_p.segments_per_cell, etc. */
typedef struct lm_segment_split_t_ {
    u8 l4_feature_segments;     /* L4: self-recurrent distal */
    u8 l4_location_segments;    /* L4: distal from L6 */
    u8 l6_location_segments;    /* L6: self-recurrent distal */
    u8 l6_feature_segments;     /* L6: distal from L4 */
    u8 l3_internal_segments;    /* L3: self-recurrent distal */
    u8 l3_external_segments;    /* L3: distal from external LMs (v1: 0) */
} lm_segment_split_t;

/* Initialize a learning module with the three layer params plus the
 * per-layer segment split and the scalar bits of L6's grid-shift
 * configuration. `pooler_cols` is the number of pooler minicolumns
 * (== L4 cols == L6 cols in this v1).
 *
 * `external_lm_outputs` for the L3 external-context stream is not yet
 * wired (v1 disables external context). */
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
);

void free_learning_module(learning_module* lm);

/* One simulation step.
 *   features                  -- pooler-driven sparse column activations
 *                                (u8 per col; non-zero => col active)
 *   movement                  -- grid-cell shift vector for L6
 *   external_output_layer_activations
 *                              -- not yet wired (v1 disables external context)
 */
void learning_module_step(
    learning_module* lm,
    features_t features,
    vec2d movement,
    lmat_u32* external_output_layer_activations
);

#endif
