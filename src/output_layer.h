#ifndef OUTPUT_LAYER_H
#define OUTPUT_LAYER_H

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>

#include "math.h"

#include "types.h"
#include "feature_layer.h"
#include "segment.h"
#include "htm.h"
#include "bitarray.h"
#include "lmat.h"

#include "lm_parameters.h"

typedef struct l3_segment_spike_cache_ {
    u8* internal_context_segments; // of size cells * internal_context_segments * sizeof(u8)
    u8* external_context_segments; // of size cells * external_context_segments * sizeof(u8)
} l3_segment_spike_cache;

typedef struct l3_segment_tensor_ {
    segment_t* feedforward; // of size cells * CONNECTIONS_PER_SEGMENT * sizeof(...)
    segment_t* internal_context; // of size cells * internal_context_segments * CONNECTIONS_PER_SEGMENT * sizeof(...)
    segment_t* external_context; // of size cells * external_context_segments * CONNECTIONS_PER_SEGMENT * sizeof(...)
} l3_segment_tensor;

typedef struct output_layer_t_ {

    l3_segment_tensor in_segments; // of shape (#cells, #segments)
    l3_segment_spike_cache spike_count_cache; // of shape (#cells, #segments)

    u32* active; // PACKED of shape (#cells)
    u32* active_prev; // PACKED of shape (#cells)

    u8* prediction_scores; // of shape (#cells)

    // used to determine top k most number of active context segments amongst the cells
    u16* prediction_score_counts; // of shape (#num_segments)

    output_layer_params_t p;
} output_layer_t;

void init_l3_segment_tensor(output_layer_t* net, feature_layer_params_t f_p, u32* seed);

void init_output_layer(output_layer_t* net, output_layer_params_t p, feature_layer_params_t f_p, u32* seed);

void output_layer_predict(output_layer_t* net, lmat_u32* external_output_layer_activations);

void output_layer_activate(output_layer_t* net, feature_layer_t* f_net, lmat_u32* external_output_layer_activations);

u32 output_layer_get_internal_context_segments_spike_count_cache_bytes(output_layer_params_t p);
u32 output_layer_get_external_context_segments_spike_count_cache_bytes(output_layer_params_t p);

u32 output_layer_get_internal_context_footprint_bytes(output_layer_params_t p);
u32 output_layer_get_external_context_footprint_bytes(output_layer_params_t p);

u32 output_layer_get_feedforward_footprint_bytes(output_layer_params_t p);

void output_layer_print_memory_footprint(output_layer_params_t p);

#endif
