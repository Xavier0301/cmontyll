#ifndef LEARNING_MODULE_H
#define LEARNING_MODULE_H

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>

#include "math.h"

#include "types.h"
#include "location.h"
#include "interfaces.h"

#include "output_layer.h"
#include "feature_layer.h"
#include "location_layer.h"

typedef struct learning_module_ {
    // layer 3
    output_layer_t output_net;

    // layer 4
    feature_layer_t feature_net;

    // layer 6
    location_layer_t location_net;
} learning_module;

void init_learning_module(
    learning_module* lm, 
    output_layer_params_t l3_params,
    feature_layer_params_t l4_params,
    location_layer_params_t l6_params,
    u32* seed
);

void learning_module_step(
    learning_module* lm,
    features_t features,
    vec2d movement,
    lmat_u32* external_output_layer_activations
);

#endif
