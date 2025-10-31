#ifndef SENSOR_MODULE_H
#define SENSOR_MODULE_H

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>

#include "math.h"

#include "types.h"
#include "tensor.h"
#include "array.h"
#include "grid_environment.h"
#include "location.h"
#include "interfaces.h"

#include "pooler.h"

typedef struct grid_sm_params_t_ {
    u32 env_min_value;
    u32 env_max_value;

    u32 encoding_length;
    u32 encoding_non_null_count;
} grid_sm_params_t;

typedef struct grid_sm_ {
    grid_sm_params_t p;

    u8* encoding_buffer;
    pooler_t pooler;
} grid_sm;

void init_sensor_module(grid_sm* sm, u32 env_min_value, u32 env_max_value, u32 num_columns);

void sensor_module(grid_sm sm, features_t* features, grid_t patch, uvec2d patch_center);

void get_point_normal_u8(vec3d* point_normal, mat_u8 depths, vec2d location);
void get_principal_curvatures_u8(i32* k1_fp, i32* k2_fp, vec3d* dir1, vec3d* dir2, mat_u8 depths, vec2d location);

void print_features(features_int_repr_t f);
void print_pose(pose_3d_repr_t p);

#endif
