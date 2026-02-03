#ifndef FEATURE_LAYER_H
#define FEATURE_LAYER_H

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>

#include "math.h"

#include "types.h"
#include "segment.h"
#include "bitarray.h"

#include "lm_parameters.h"

typedef struct location_layer_params_t_ location_layer_params_t;
typedef struct location_layer_t_ location_layer_t;

typedef struct l4_segment_spike_cache_ {
    u8* location_segments; // of size cols * cells * location_segments * sizeof(u8)
    u8* feature_segments; // of size cols * cells * feature_segments * sizeof(u8)
} l4_segment_spike_cache;

typedef struct l4_segment_tensor_ {
    // the feedforward segments are already taken care of in the pooling layer, transmitted via "active_columns" in activate()
    segment_t* feature_context; // of size cols * cells * feature_segments * CONNECTIONS_PER_SEGMENT * sizeof(segment_data_t)
    segment_t* location_context; // of size cols * cells * location_segments * CONNECTIONS_PER_SEGMENT * sizeof(segment_data_t)
} l4_segment_tensor;

typedef struct feature_layer_t_ {
    l4_segment_tensor in_segments; // of shape (#cols, #cells, #segments)
    l4_segment_spike_cache spike_count_cache; // of shape (#cols, #cells, #segments)

    u32* predicted; // of shape (#minicols) where each entry is a bitarray representing the cells in each col [sparse x sparse]
    u32* active; // of shape (#minicols) where each entry is a bitarray representing the cells in each col [sparse x sparse]

    u32* active_prev; // same as active. Represents the last state, used for learning

    feature_layer_params_t p;
} feature_layer_t;

void init_l4_segment_tensor(feature_layer_t* net, location_layer_params_t l_p, u32* seed);

void init_feature_layer(feature_layer_t* net, feature_layer_params_t p, location_layer_params_t l_p, u32* seed);

void feature_layer_predict(feature_layer_t* net, location_layer_t* l_net);

void feature_layer_activate(feature_layer_t* net, u8* active_columns, location_layer_t* l_net);

u32 feature_layer_get_feature_segments_spike_count_cache_bytes(feature_layer_params_t p);
u32 feature_layer_get_location_segments_spike_count_cache_bytes(feature_layer_params_t p);

u32 feature_layer_get_feature_context_footprint_bytes(feature_layer_params_t p);
u32 feature_layer_get_location_context_footprint_bytes(feature_layer_params_t p);

void feature_layer_print_memory_footprint(feature_layer_params_t p);

#endif
