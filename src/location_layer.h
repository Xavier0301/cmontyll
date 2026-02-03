#ifndef LOCATION_LAYER_H
#define LOCATION_LAYER_H

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>

#include "math.h"

#include "types.h"
#include "location.h"
#include "distributions.h"
#include "segment.h"
#include "htm.h"

typedef struct feature_layer_params_t_ feature_layer_params_t;
typedef struct feature_layer_t_ feature_layer_t;

#include "lm_parameters.h"

typedef struct l6_segment_spike_cache_ {
    u8* location_segments; // of size cols * cells * location_segments * sizeof(u8)
    u8* feature_segments; // of size cols * cells * feature_segments * sizeof(u8)
} l6_segment_spike_cache;

typedef struct l6_segment_tensor_ {
    // the feedforward segments are already taken care of in the pooling layer, transmitted via "active_columns" in activate()
    segment_t* location_context; // of size cols * cells * location_segments * CONNECTIONS_PER_SEGMENT * sizeof(segment_data_t)
    segment_t* feature_context; // of size cols * cells * feature_segments * CONNECTIONS_PER_SEGMENT * sizeof(segment_data_t)
} l6_segment_tensor;

typedef struct location_layer_t_ {

    l6_segment_tensor in_segments; // of shape (#cols, #cells, #segments, #connections)
    l6_segment_spike_cache spike_count_cache; // of shape (#cols, #cells, #segments)

    u32* active; // bitarray of shape cols
    u32* predicted; // bitarray of shape cols

    u32* active_prev; // bitarray of shape cols

    location_layer_params_t p;
} location_layer_t;

void init_l6_segment_tensor(location_layer_t* net, feature_layer_params_t f_p, u32* seed);

void init_location_layer(location_layer_t* net, location_layer_params_t p, feature_layer_params_t f_p, u32* seed);

void location_layer_predict(location_layer_t* net, feature_layer_t* f);

// shift grid cell activity following movement and make location unique according to predictions
void location_layer_activate(location_layer_t* net, vec2d movement, feature_layer_t* f_net);

u32 location_layer_get_location_segments_spike_count_cache_bytes(location_layer_params_t p);
u32 location_layer_get_feature_segments_spike_count_cache_bytes(location_layer_params_t p);

u32 location_layer_get_location_context_footprint_bytes(location_layer_params_t p);
u32 location_layer_get_feature_context_footprint_bytes(location_layer_params_t p);

void location_layer_print_memory_footprint(location_layer_params_t p);

#endif
