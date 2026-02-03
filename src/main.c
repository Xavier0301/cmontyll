#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include <time.h>

#include <unistd.h> // For sleep function

#include "grid_environment.h"
#include "sensor_module.h"
#include "learning_module.h"
#include "motor_policy.h"

int main(int argc, char *argv[]) {  
    u32 num_steps = 0;

    const char error_message[] = "Error: usage: %s X\n\t \
        X is the number of steps (integer)\n";

    // argv[0] is program name, argv[1] is the argument
    if (argc < 2) {
        printf(error_message, argv[0]);
        exit(1);
    }

    num_steps = atoi(argv[1]);

    if (num_steps <= 0) {
        printf("Error: Number of steps must be a positive integer.\n");
        exit(1);
    }

    printf("Running simulation with %d steps.\n", num_steps);

    u32 seed = 5;

    grid_t env;
    u32 env_sidelen = 10;
    init_grid_env(&env, env_sidelen, env_sidelen);
    populate_grid_env_random(&env, &seed);

    grid_t patch; // grid_t is an ugly abstraction for the patch
    u32 patch_sidelen = 3;
    init_grid_env(&patch, patch_sidelen, patch_sidelen);
    uvec2d patch_center = (uvec2d) { .x = patch_sidelen / 2, .y = patch_sidelen / 2 };

    bounds_t bounds = get_bounds(env_sidelen, env_sidelen, patch_sidelen, patch_sidelen);

    uvec2d agent_location = { .x = 5, .y = 1 }; // start location

    u32 num_cols = 1024;

    grid_sm sm;
    init_sensor_module(&sm, GRID_ENV_MIN_VALUE, GRID_ENV_MAX_VALUE, num_cols, &seed);

    print_pooler(&sm.pooler);

    random_motor_policy_t motor_policy;
    init_random_motor_policy(&motor_policy, agent_location, bounds, num_steps, &seed);

    features_t f;
    init_features(&f, sm.pooler.params.num_minicols, sm.pooler.params.top_k);

    htm_params_t htm_params = (htm_params_t) {
        .permanence_threshold = REPR_u8(0.5),
        .segment_spiking_threshold = 15,

        .perm_increment = REPR_u8(0.06f),
        .perm_decrement = REPR_u8(0.04f),
        .perm_decay = 1 // 1/256, the smallest possible non-zero decay
    };

    extended_htm_params_t ext_htm_params = (extended_htm_params_t) {
        .feedforward_permanence_threshold = REPR_u8(0.5),
        .context_permanence_threshold = REPR_u8(0.5),

        .feedforward_activation_threshold = 3,
        .context_activation_threshold = 18,

        .min_active_cells = 10,
    };
 
    learning_module lm;
    init_learning_module(
        &lm, 
        (output_layer_params_t) {
            .cells = 1024, // cells per col
            .log_cells = 10,

            .internal_context_segments = 12,
            .external_context_segments = 0,

            .external_cells = 1024, // external output dim
            .external_lms = 0, // num of connected external lms
            
            .htm = htm_params,
            .extended_htm = ext_htm_params
        }, 
        (feature_layer_params_t) {
            .cols = num_cols,
            .cells = 8, // cells per col

            .feature_segments = 6,
            .location_segments = 6,

            .htm = htm_params
        },
        (location_layer_params_t) {
            .cols = 1024,
            .log_cols_sqrt = (u32) log2(sqrt(1024)), // 5
            .cells = 8,

            .location_segments = 6,
            .feature_segments = 6,

            .log_scale = (uvec2d) { .x = 0, .y = 0 },

            .htm = htm_params
        },
        &seed
    );

    vec2d movement = { .x = 0, .y = 0 };

    printf("\n");
    feature_layer_print_params(lm.feature_net.p);
    feature_layer_print_memory_footprint(lm.feature_net.p);
    printf("\n");
    location_layer_print_params(lm.location_net.p);
    location_layer_print_memory_footprint(lm.location_net.p);
    printf("\n");
    output_layer_print_params(lm.output_net.p);
    output_layer_print_memory_footprint(lm.output_net.p);
    printf("\n");

    print_grid(&env);

    clock_t start = clock();

    for(u32 step = 0; step < num_steps; ++step) {
        extract_patch(&patch, &env, agent_location, patch_sidelen);

        sensor_module(sm, &f, patch, patch_center);

        learning_module_step(&lm, f, movement, NULL);


        movement = random_motor_policy(&motor_policy, f);

        agent_location.x += movement.x;
        agent_location.y += movement.y;

#if PRINT == 2
        printf("--- step %u: agent at location (%u, %u)\n", step, agent_location.x, agent_location.y);

        print_grid(&patch);

        printf("column activation sparsity: ");
        print_spvec_u8(f.active_columns, f.num_columns);

        print_packed_spvec_u32(lm.output_net.active, lm.output_net.p.cells);

        printf("\n");      
#endif
    }

    clock_t end = clock();
    double total_time = ((double)(end - start)) / CLOCKS_PER_SEC;

    fprintf(stderr, "\ntime per step: %.3f ms\n", total_time * 1000 / num_steps);

    return 0;
}
