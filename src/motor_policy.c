#include "motor_policy.h"

#include "stdlib.h"
#include "distributions.h"

void init_random_motor_policy(random_motor_policy_t* policy, uvec2d start_location, bounds_t bounds, u32 steps, u32* seed) {
    policy->pregenerated_movements = calloc(steps, sizeof(*policy->pregenerated_movements));
    policy->current_step = 0;
    policy->num_steps = steps;

    reset_random_motor_policy(policy, start_location, bounds, steps, seed);
}

void reset_random_motor_policy(random_motor_policy_t* policy, uvec2d start_location, bounds_t bounds, u32 steps, u32* seed) {
    uvec2d last_location = { .x = start_location.x, .y = start_location.y };
    uvec2d next_location;
    for(u32 i = 0; i < steps; ++i) {
        next_location.x = unif_rand_range_u32(bounds.min_x, bounds.max_x, seed);
        next_location.y = unif_rand_range_u32(bounds.min_y, bounds.max_y, seed);

        vec2d movement;
        // movement.x = (i32) next_location.x - (i32) last_location.x;
        // movement.y = (i32) next_location.y - (i32) last_location.y;
        movement.x = 0;
        movement.y = 0;

        if(movement.x >= 300 || movement.y >= 300 || movement.x <= -300 || movement.y <= -300) {
            printf("CRITICAL ERROR: crazy movement, %d, %d\n", 
               movement.x, movement.y);
            abort();
        }

        // printf("movement %u is (%d, %d) between (%u, %u) and (%u, %u)\n", i, movement.x, movement.y, last_location.x, last_location.y, next_location.x, next_location.y);

        policy->pregenerated_movements[i] = movement;

        last_location.x = next_location.x;
        last_location.y = next_location.y;
    }
}

/**
 * @brief 
 * 
 * @returns vec2d representing the movement
 * 
 * @param features ignored in this policy, here for the signature
 * @param pose ignored in this policy, here for the signature
 */
vec2d random_motor_policy(random_motor_policy_t* policy, features_t features) {
    (void) features;

    if(policy->current_step >= policy->num_steps) {
        printf("CRITICAL ERROR: Step %u requested, but only %u allocated!\n", 
               policy->current_step, policy->num_steps);
        abort();
    }

    vec2d movement = policy->pregenerated_movements[policy->current_step];
    policy->current_step += 1;

    return movement;
}
