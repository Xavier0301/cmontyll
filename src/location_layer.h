#ifndef LOCATION_LAYER_H
#define LOCATION_LAYER_H

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
#include "segment.h"

#include "location_module.h"

#include "bitarray.h"

typedef struct location_layer_params_t_ {
    u32 modules;

    location_module_params_t* module_params;
} location_layer_params_t;

typedef struct location_layer_t_ {
    location_module_t* modules;

    location_layer_params_t p;
} location_layer_t;

void init_location_layer(location_layer_t* net, location_layer_params_t l6_params);

void location_layer_activate(location_layer_t* net, uvec2d location);

void location_layer_move(location_layer_t* net, vec2d movement);

static inline u32 location_layer_check_active(location_layer_t* net, location_index index) {
    return location_module_check_active(net->modules[index.module], index.cell_x, index.cell_y);
}

static inline u32 location_layer_check_active_prev(location_layer_t* net, location_index index) {
    return location_module_check_active_prev(net->modules[index.module], index.cell_x, index.cell_y);
}

#endif
