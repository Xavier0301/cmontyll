#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h> 

#include <omp.h>

#include "grid_environment.h"
#include "sensor_module.h"
#include "learning_module.h"
#include "motor_policy.h"

#define LOG_OUT_CELLS (10) // 2^10 = 1024
#define OUT_CELLS (1 << LOG_OUT_CELLS)
#define OUT_CELL_BYTES (OUT_CELLS >> 3) // each cell can be coded on 1/8th=1/2^3th of a byte
#define OUT_CELL_WORDS (OUT_CELL_BYTES >> 2) // word=u32 i.e. 4=2^2 bytes per word

#ifndef NUM_NEIGHBORS
#define NUM_NEIGHBORS (20) 
#endif

int main(int argc, char *argv[]) {  
    u32 num_steps = 0;

    const char error_message[] = "Error: usage: %s X\n\t \
        X is the number of steps (integer)\n";

    // argv[0] is program name, argv[1] is the argument
    if (argc < 2) {
        printf(error_message, argv[0]);
        exit(1);
    }

    /* Convert string argument to int */
    num_steps = atoi(argv[1]);

    if (num_steps < 0) {
        printf("Error: Number of steps must be a positive integer.\n");
        exit(1);
    }

    printf("Running simulation with %d steps.\n", num_steps);

    // ---------------------------------------------------------
    // 1. SETUP SHARED RESOURCES
    // ---------------------------------------------------------

    u32 shared_seed = 213;
    
    grid_t env;
    u32 env_sidelen = 10;
    init_grid_env(&env, env_sidelen, env_sidelen);
    populate_grid_env_random(&env, &shared_seed);

    u32 num_cols = 1024;

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

    output_layer_params_t output_layer_params = (output_layer_params_t) {
        .cells = OUT_CELLS, // cells per col
        .log_cells = LOG_OUT_CELLS,

        .internal_context_segments = 6,
        .external_context_segments = 6,

        .external_cells = OUT_CELLS, // external output dim
        .external_lms = NUM_NEIGHBORS, // num of connected external lms
        
        .htm = htm_params,
        .extended_htm = ext_htm_params
    };

    feature_layer_params_t feature_layer_params = (feature_layer_params_t) {
        .cols = num_cols,
        .cells = 8, // cells per col

        .feature_segments = 6,
        .location_segments = 6,

        .htm = htm_params
    };

    location_layer_params_t location_layer_params = (location_layer_params_t) {
        .cols = 1024,
        .log_cols_sqrt = (u32) log2(sqrt(1024)), // 5
        .cells = 8,

        .location_segments = 6,
        .feature_segments = 6,

        .log_scale = (uvec2d) { .x = 0, .y = 0 },

        .htm = htm_params
    };

    int max_threads = omp_get_max_threads(); 
    
    lmat_u32 global_exchange_buffer;
    lmat_u32_init(&global_exchange_buffer, max_threads, LOG_OUT_CELLS - 5);

    printf("Configured Max (OpenMP limit): %d\n", omp_get_max_threads()); 
    printf("Hardware Cores: %d\n", omp_get_num_procs());                
    printf("Current Active Threads: %d\n", omp_get_num_threads());  

    printf("Allocated Global Exchange Buffer: %lu KB\n",
           (max_threads * OUT_CELL_WORDS * sizeof(u32)) / 1024);

    // ---------------------------------------------------------
    // 2. PARALLEL REGION
    // ---------------------------------------------------------
    double start_time;

    #pragma omp parallel 
    {
        int tid = omp_get_thread_num();
        int num_threads = omp_get_num_threads();

        u32 tseed = tid;

        if(num_threads >= 20) {
            usleep(tid * 100000);
        }

        grid_t patch; // grid_t is an ugly abstraction for the patch
        u32 patch_sidelen = 3;
        init_grid_env(&patch, patch_sidelen, patch_sidelen);
        uvec2d patch_center = (uvec2d) { .x = patch_sidelen / 2, .y = patch_sidelen / 2 };

        bounds_t bounds = get_bounds(env_sidelen, env_sidelen, patch_sidelen, patch_sidelen);

        uvec2d agent_location = { .x = 5, .y = 1 }; // start location

        grid_sm sm;
        init_sensor_module(&sm, GRID_ENV_MIN_VALUE, GRID_ENV_MAX_VALUE, num_cols, &tseed);

        print_pooler(&sm.pooler);

        random_motor_policy_t motor_policy;
        init_random_motor_policy(&motor_policy, agent_location, bounds, num_steps, &tseed);

        features_t f;
        init_features(&f, sm.pooler.params.num_minicols, sm.pooler.params.top_k);
        
        learning_module lm;
        init_learning_module(
            &lm, 
            output_layer_params, 
            feature_layer_params,
            location_layer_params,
            &tseed
        );

        // --- EXTERNAL OUTPUT SETUP ---
        lmat_u32 external_output_layer_activations; // >> 3 for packed in u8, >> 2 for number of u32
        lmat_u32_init(&external_output_layer_activations, NUM_NEIGHBORS, LOG_OUT_CELLS - 5); 
        
        // Random topology setup
        u32 incident_lm_tids[NUM_NEIGHBORS];
        for(u32 i = 0; i < NUM_NEIGHBORS; ++i)
            incident_lm_tids[i] = unif_rand_range_u32_except(0, num_threads - 1, tid, &tseed);

        vec2d movement = { .x = 0, .y = 0 };

        #pragma omp barrier 

        #pragma omp master
        start_time = omp_get_wtime();

        for(u32 step = 0; step < num_steps; ++step) {            
            extract_patch(&patch, &env, agent_location, patch_sidelen);
            sensor_module(sm, &f, patch, patch_center);

            for(u32 i = 0; i < NUM_NEIGHBORS; ++i) {
                memcpy(
                    LMATP(external_output_layer_activations, i, 0), // dst
                    LMATP(global_exchange_buffer, incident_lm_tids[i], 0), // src
                    OUT_CELL_BYTES // bytes
                );
            }
            learning_module_step(&lm, f, movement, &external_output_layer_activations);

            movement = random_motor_policy(&motor_policy, f);
            agent_location.x += movement.x;
            agent_location.y += movement.y;

            memcpy(
                LMATP(global_exchange_buffer, tid, 0), // dst
                lm.output_net.active, // src
                OUT_CELL_BYTES // bytes
            );

            #pragma omp barrier
        }
    } 
    double end_time = omp_get_wtime();
    double total_time = end_time - start_time ;
    fprintf(stderr, "\nOMP time per step: %.3f ms\n", total_time * 1000 / num_steps);
    fflush(stderr);

    // Cleanup Global Buffer
    // free(global_exchange_buffer.data);

    return 0;
}
