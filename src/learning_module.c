#include "learning_module.h"

/**
 * Learning modules create a sensorimotor model of the objects/environment they learn
 * 
 * Our implementation relies on an layer4(features)/layer6(location) network 
 * implemented with an htm networks and grid cell modules
*/

void init_learning_module(
    learning_module* lm, 
    output_layer_params_t l3_params,
    feature_layer_params_t l4_params,
    location_layer_params_t l6_params
) {
    init_output_layer(&lm->output_net, l3_params, l4_params);
    init_feature_layer(&lm->feature_net, l4_params, l6_params);
    init_location_layer(&lm->location_net, l6_params);
}

void learning_module_step(learning_module* lm, features_t features, vec2d movement) {
    location_layer_move(&lm->location_net, movement);

    feature_layer_predict(&lm->feature_net, &lm->location_net);
    feature_layer_activate(&lm->feature_net, features.active_columns, &lm->location_net);

    output_layer_activate(&lm->output_net, &lm->feature_net);
}
