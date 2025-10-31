#include "location_module.h"

void init_location_module(location_module_t* m, location_module_params_t p) {
    m->active_cells.length = 1;
    m->active_cells.data = calloc(1, sizeof(*m->active_cells.data));

    m->active_cells_prev.length = 1;
    m->active_cells_prev.data = calloc(1, sizeof(*m->active_cells_prev.data));
}

void location_module_activate(location_module_t m, location_module_params_t p, uvec2d location) {
    uvec2d l = (uvec2d) { 
        .x = location.x >> p.log_scale.x, 
        .y = location.y >> p.log_scale.y 
    };
    
    m.active_cells.data[0] = (coo_entry_u1) { 
        .row = (l.x + p.bias.x) % p.cells_sqrt, 
        .col = (l.y + p.bias.y) % p.cells_sqrt 
    };
}

void location_module_move(location_module_t m, location_module_params_t p, vec2d movement) {
    vec2d mvmt = (vec2d) { .x = movement.x >> p.log_scale.x, .y = movement.y >> p.log_scale.y };
    
    for(u32 i = 0; i < m.active_cells.length; ++i) {
        coo_entry_u1 prev_cell = m.active_cells.data[i];
        m.active_cells.data[i] = (coo_entry_u1) { 
            .row = (prev_cell.row + mvmt.x) % p.cells_sqrt, 
            .col = (prev_cell.col + mvmt.y) % p.cells_sqrt
        };

        m.active_cells_prev.data[i] = prev_cell;
    }
}
