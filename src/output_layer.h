#ifndef OUTPUT_LAYER_H
#define OUTPUT_LAYER_H

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>

#include "math.h"

#include "types.h"
#include "tensor.h"
#include "array.h"
#include "location.h"
#include "interfaces.h"
#include "sparse.h"
#include "feature_layer.h"
#include "segment.h"

#include "bitarray.h"

typedef struct output_layer_params_t_ {
    u16 cells;

    u8 internal_context_segments;
    u8 external_context_segments;
    u8 context_segments;

    u8 feedforward_permanence_threshold;
    u8 context_permanence_threshold;

    u8 feedforward_activation_threshold;
    u8 context_activation_threshold;

    u8 min_active_cells;

    u8 segment_spiking_threshold;

    u8 perm_increment;
    u8 perm_decrement;
    u8 perm_decay; // how much to decay predicted but inactive cells 
    // TODO: maybe add a decay period?
} output_layer_params_t;


typedef struct l3_segment_tensor_ {
    segment_t* feedforward; // of size cells * CONNECTIONS_PER_SEGMENT * sizeof(...)
    segment_t* context; // of size cells * context_segments * CONNECTIONS_PER_SEGMENT * sizeof(...)
} l3_segment_tensor;

typedef struct output_layer_t_ {

    l3_segment_tensor in_segments; // of shape (#cells, #segments)

    u32* active; // PACKED of shape (#cells)
    u32* active_prev; // PACKED of shape (#cells)

    u8* prediction_scores; // of shape (#cells)

    // used to determine top k most number of active context segments amongst the cells
    u16* prediction_score_counts; // of shape (#num_segments)

    output_layer_params_t p;
} output_layer_t;

void init_l3_segment_tensor(output_layer_t* net, feature_layer_params_t feature_layer_p);

void init_output_layer(output_layer_t* net, output_layer_params_t p, feature_layer_params_t feature_layer_p);

void output_layer_predict(output_layer_t* net);

void output_layer_activate(output_layer_t* net, feature_layer_t* feature_net);

#endif
