#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include <unistd.h> // For sleep function

#include "grid_environment.h"
#include "sensor_module.h"
#include "learning_module.h"
#include "motor_policy.h"

int main(int argc, char *argv[]) {  

// Put the error message in a char array:
    const char error_message[] = "Error: usage: %s X\n\t \
        0 is for training from scratch\n";


    /* Error Checking */
    if(argc < 0) {
        printf(error_message, argv[0]);
        exit(1);
    }

    u32 num_step = 10;

    grid_t env;
    u32 env_sidelen = 10;
    init_grid_env(&env, env_sidelen, env_sidelen);
    populate_grid_env_random(&env);

    grid_t patch; // grid_t is an ugly abstraction for the patch
    u32 patch_sidelen = 3;
    init_grid_env(&patch, patch_sidelen, patch_sidelen);
    uvec2d patch_center = (uvec2d) { .x = patch_sidelen / 2, .y = patch_sidelen / 2 };

    bounds_t bounds = get_bounds(env_sidelen, env_sidelen, patch_sidelen, patch_sidelen);

    uvec2d agent_location = { .x = 5, .y = 1 }; // start location

    u32 num_cols = 1024;

    grid_sm sm;
    init_sensor_module(&sm, GRID_ENV_MIN_VALUE, GRID_ENV_MAX_VALUE, num_cols);

    print_pooler(&sm.pooler);

    random_motor_policy_t motor_policy;
    init_random_motor_policy(&motor_policy, agent_location, bounds, num_step);

    features_t f;
    init_features(&f, sm.pooler.params.num_minicols, sm.pooler.params.top_k);

    //     u16 cols;
    // u16 log_cells_per_col; // cells_per_col has to be a power of 2, also <= 32 i.e. log(cells_per) is max 5
    // u8 segments;

    // u8 permanence_threshold;
    // u8 segment_depolarization_threshold;

    // u8 perm_increment;
    // u8 perm_decrement;
    // u8 perm_decay;
    learning_module lm;
    init_learning_module(
        &lm, 
        (output_layer_params_t) {
            .cells = 1024, // cells per col

            .internal_context_segments = 10,
            .external_context_segments = 0,
            .context_segments = 10, // segments per cell

            .feedforward_permanence_threshold = REPR_u8(0.5),
            .context_permanence_threshold = REPR_u8(0.5),
            .segment_spiking_threshold = 15,

            .feedforward_activation_threshold = 3,
            .context_activation_threshold = 18,

            .min_active_cells = 10,

            .perm_increment = REPR_u8(0.06f),
            .perm_decrement = REPR_u8(0.04f),
            .perm_decay = 1 // 1/256, the smallest possible non-zero decay
        }, 
        (feature_layer_params_t) {
            .cols = num_cols,
            .cells = 10, // cells per col

            .feature_segments = 5,
            .location_segments = 5,
            .segments = 10, // segments per cell

            .permanence_threshold = REPR_u8(0.5),
            .segment_spiking_threshold = 15,

            .perm_increment = REPR_u8(0.06f),
            .perm_decrement = REPR_u8(0.04f),
            .perm_decay = 1 // 1/256, the smallest possible non-zero decay
        },
        (location_layer_params_t) {
            .modules = 10,
            .module_params = calloc(10, sizeof(location_module_params_t))
        }
    );

    vec2d movement = { .x = 0, .y = 0 };
    location_layer_activate(&lm.location_net, agent_location); 

    print_grid(&env);

    for(u32 step = 0; step < num_step; ++step) {
        printf("--- step %u: agent at location (%u, %u)\n", step, agent_location.x, agent_location.y);
        extract_patch(&patch, &env, agent_location, patch_sidelen);

        print_grid(&patch);

        sensor_module(sm, &f, patch, patch_center);

        printf("column activation sparsity: ");
        print_spvec_u8(f.active_columns, f.num_columns);

        learning_module_step(&lm, f, movement);

        print_packed_spvec_u32(lm.output_net.active, lm.output_net.p.cells);

        movement = random_motor_policy(&motor_policy, f);

        agent_location.x += movement.x;
        agent_location.y += movement.y;

        printf("\n");
    }

    return 0;
}
