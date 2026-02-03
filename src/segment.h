#ifndef SEGMENT_H
#define SEGMENT_H

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>

#include "math.h"

#include "types.h"


// ----- OUTPUT LAYER -----

// u24
typedef struct __attribute__((packed)) internal_output_index_ {
    u16 cell;
    u8 filler; // unused
} internal_output_index;

// u24
typedef struct __attribute__((packed)) external_output_index_ {
    u16 cell;
    u8 lm_id; // learning module id
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
    u16 col;
    u8 cell;
} location_index;

// ----- ABSTRACT SEGMENT -----

union cell_index {
    internal_output_index internal_output;
    external_output_index external_output;
    feature_index feature;
    location_index location;
};

enum segment_index_type {
    INTERNAL_OUTPUT_INDEX_TYPE,
    EXTERNAL_OUTPUT_INDEX_TYPE,
    FEATURE_INDEX_TYPE,
    LOCATION_INDEX_TYPE
};

// u32
typedef struct __attribute__((packed)) segment_data_ {
    union cell_index index;
    u8 permanence;
} segment_data;

#define CONNECTIONS_PER_SEGMENT 40

typedef struct __attribute__((aligned(8))) segment_t_ {
    segment_data connections[CONNECTIONS_PER_SEGMENT]; // each connections takes 4 bytes, we "cast" it to the proper x_segment_data in the code
    u8 num_connections;
    // u8 connection_count; // used for learning, to know which segment was spiking
} segment_t;

#endif
