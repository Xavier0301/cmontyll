#ifndef LOCATION_MODULE_H
#define LOCATION_MODULE_H

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

typedef struct location_module_params_t_ {
    uvec2d log_scale;
    uvec2d bias;
    u32 cells_sqrt;
} location_module_params_t;

typedef struct location_module_t_ {
    coo_u1 active_cells;
    coo_u1 active_cells_prev; // Represents the last state, used for learning
} location_module_t;

void init_location_module(location_module_t* m, location_module_params_t p);

void location_module_activate(location_module_t m, location_module_params_t p, uvec2d location);

void location_module_move(location_module_t m, location_module_params_t p, vec2d movement);

static inline u32 location_module_check_active(location_module_t m, u8 cell_index_x, u8 cell_index_y) {
    for(u32 i = 0; i < m.active_cells.length; ++i) {
        coo_entry_u1 cell = m.active_cells.data[i];
        if(cell.row == cell_index_x && cell.col == cell_index_y) return 1;
    }

    return 0;
}

static inline u32 location_module_check_active_prev(location_module_t m, u8 cell_index_x, u8 cell_index_y) {
    for(u32 i = 0; i < m.active_cells_prev.length; ++i) {
        coo_entry_u1 cell = m.active_cells_prev.data[i];
        if(cell.row == cell_index_x && cell.col == cell_index_y) return 1;
    }

    return 0;
}


#endif
