#ifndef LAYER_H
#define LAYER_H

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include "string.h"

#include "math.h"

#include "types.h"
#include "tensor.h"
#include "distributions.h"

#include "htm.h"

typedef struct htm_params_t_ htm_params_t;

#include "lm_parameters.h"

typedef struct out_projection_t_ {
    u8* out_accumulator_pointer;
    u8 permanence;
} out_projection_t;

DEFINE_TENSOR_STRUCT_WNAME(out_projection_t, tnsr_proj);

typedef struct layer_params_t_ {
    u16 cols;
    u16 cells; // per columns
    u8 segments; // per cell

    u16 projections; // per cell

    htm_params_t htm;
} layer_params_t;

typedef struct layer_t_ {

    tnsr_proj out_projections; // of shape (#cols, #cells, #projections)

    tnsr_u8 segment_accumulators; // of shape (#cols, #cells, #segments)

    u32* active; // bitarray of shape cols
    u32* predicted; // bitarray of shape cols

    u32* active_prev; // bitarray of shape cols

    layer_params_t p;
} layer_t;

void layer_init_connections(layer_t* net, layer_params_t p, u32* seed);

void layer_init_state(layer_t* net, layer_params_t p, u32* seed);

void layer_predict(layer_t* net);

void layer_activate(layer_t* net);
 
void layer_learn(layer_t* net);

u32 layer_get_connections_footprint_bytes(layer_params_t p);
u32 layer_get_accumulators_footprint_bytes(layer_params_t p);

void layer_print_memory_footprint(layer_params_t p);

#endif // LAYER_H
