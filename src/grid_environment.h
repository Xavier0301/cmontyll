#ifndef GRID_ENVIRONMENT_H
#define GRID_ENVIRONMENT_H

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>

#include "math.h"

#include "types.h"
#include "tensor.h"
#include "location.h"
#include "bounds.h"

#define GRID_ENV_MIN_VALUE 0
#define GRID_ENV_MAX_VALUE 4

#define GRID_ENV_MIN_DEPTH 10
#define GRID_ENV_MAX_DEPTH 50

typedef struct grid_t_ {
    mat_u32 values;
    mat_u8 depths;

    u32 rows;
    u32 cols;
} grid_t;

void init_grid_env(grid_t* env, u32 rows, u32 cols);
void populate_grid_env_random(grid_t* env, u32* seed);

bounds_t get_bounds(u32 env_size_x, u32 env_size_y, u32 patch_size_x, u32 patch_size_y);

void extract_patch(grid_t* patch, grid_t* env, uvec2d location, u32 patch_sidelen);

void print_grid(grid_t* env);


#endif
