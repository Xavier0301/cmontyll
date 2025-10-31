#ifndef SEGMENT_H
#define SEGMENT_H

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>

#include "math.h"

#include "types.h"
#include "tensor.h"
#include "array.h"
#include "location.h"
#include "interfaces.h"
#include "sparse.h"

#include "bitarray.h"

// ----- OUTPUT LAYER -----

// u24
typedef struct __attribute__((packed)) internal_output_index_ {
    u16 cell;
    u8 filler; // unused
} internal_output_index;

// u24
typedef struct __attribute__((packed)) external_output_index_ {
    u16 cell;
    u8 learning_module_id;
} external_output_index;

// ----- FEATURE LAYER -----

// u24
typedef struct __attribute__((packed)) feature_index_ {
    u16 col;
    u8 cell;
} feature_index;

// ----- LOCATION LAYER -----

// u24
typedef struct __attribute__((packed)) location_index_ {
    u8 module;
    u8 cell_x;
    u8 cell_y;
} location_index;

// ----- ABSTRACT SEGMENT -----

union cell_index {
    internal_output_index internal_output;
    external_output_index external_output;
    feature_index feature;
    location_index location;
};

// u32
typedef struct __attribute__((packed)) segment_data_ {
    union cell_index index;
    u8 permanence;
} segment_data;

#define CONNECTIONS_PER_SEGMENT 40

typedef struct segment_t_ {
    segment_data connections[CONNECTIONS_PER_SEGMENT]; // each connections takes 4 bytes, we "cast" it to the proper x_segment_data in the code
    u8 num_connections;
    u8 connection_count; // used for learning, to know which segment was spiking
} segment_t;

#endif
