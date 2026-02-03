#ifndef MOTOR_POLICY_H
#define MOTOR_POLICY_H

#include "location.h"
#include "interfaces.h"
#include "bounds.h"

typedef struct random_motor_policy_t_ {
    vec2d* pregenerated_movements;

    u32 current_step;
    u32 num_steps;
} random_motor_policy_t;

void init_random_motor_policy(random_motor_policy_t* policy, uvec2d start_location, bounds_t bounds, u32 steps, u32* seed);
void reset_random_motor_policy(random_motor_policy_t* policy, uvec2d start_location, bounds_t bounds, u32 steps, u32* seed);

vec2d random_motor_policy(random_motor_policy_t* policy, features_t features);

#endif
