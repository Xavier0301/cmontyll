#include "location_layer.h"

#include "distributions.h"

/**
 * @brief 
 * 
 * @param net 
 * @param l6_params .module_params is assumed to be nonsense but allocated, it is our job to populate it
 */
void init_location_layer(location_layer_t* net, location_layer_params_t l6_params) {
    net->p = l6_params;

    net->modules = calloc(net->p.modules, sizeof(*net->modules));
    for(u32 m = 0; m < net->p.modules; ++m) {
        location_module_params_t p;

        p.cells_sqrt = unif_rand_range_u32(5, 13);
        p.bias = (uvec2d) {
            .x = unif_rand_u32(p.cells_sqrt - 1),
            .y = unif_rand_u32(p.cells_sqrt - 1)
        };
        p.log_scale = (uvec2d) {
            .x = 0,
            .y = 0
        };

        net->p.module_params[m] = p;
        
        init_location_module(&net->modules[m], net->p.module_params[m]);
    }
}

void location_layer_activate(location_layer_t* net, uvec2d location) {
    for(u32 m = 0; m < net->p.modules; ++m) 
        location_module_activate(net->modules[m], net->p.module_params[m], location);
}

void location_layer_move(location_layer_t* net, vec2d movement) {
    for(u32 m = 0; m < net->p.modules; ++m) 
        location_module_move(net->modules[m], net->p.module_params[m], movement);
}
