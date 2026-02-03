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
    location_layer_params_t l6_params,
    u32* seed
) {
    init_output_layer(&lm->output_net, l3_params, l4_params, seed);
    init_feature_layer(&lm->feature_net, l4_params, l6_params, seed);
    init_location_layer(&lm->location_net, l6_params, l4_params, seed);
}

void learning_module_step(learning_module* lm, features_t features, vec2d movement, lmat_u32* external_output_layer_activations) {
    location_layer_predict(&lm->location_net, &lm->feature_net);
    location_layer_activate(&lm->location_net, movement, &lm->feature_net);

    feature_layer_predict(&lm->feature_net, &lm->location_net);
    feature_layer_activate(&lm->feature_net, features.active_columns, &lm->location_net);

    output_layer_predict(&lm->output_net, external_output_layer_activations);
    output_layer_activate(&lm->output_net, &lm->feature_net, external_output_layer_activations);
}
