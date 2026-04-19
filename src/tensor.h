#ifndef TNSR_H
#define TNSR_H

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>

#include "math.h"

#include "types.h"


#define TENSOR_TYPE_(symbol) tnsr_##symbol##_
#define TENSOR_TYPE(symbol) tnsr_##symbol

// stride2 = shape3
#define DEFINE_TENSOR_STRUCT(symbol) \
    typedef struct TENSOR_TYPE_(symbol) { \
        u32 stride1; \
        u32 shape1; \
        u32 shape2; \
        u32 shape3; \
        symbol* data; \
    } TENSOR_TYPE(symbol)

DEFINE_TENSOR_STRUCT(u32);
DEFINE_TENSOR_STRUCT(u16);
DEFINE_TENSOR_STRUCT(u8);

#define DEFINE_TENSOR_STRUCT_WNAME(symbol, name) \
    typedef struct name##_ { \
        u32 stride1; \
        u32 shape1; \
        u32 shape2; \
        u32 shape3; \
        symbol* data; \
    } name

#define TNSR(t, i, j, k) ((t).data[(i) * (t).stride1 + (j) * (t).shape3 + (k)])
#define TNSR_P(t, i, j, k) ((t).data + (i) * (t).stride1 + (j) * (t).shape3 + (k))

#define TENSOR_INIT(t, shape1_, shape2_, shape3_, type) \
    do { \
        (t)->stride1 = shape2_ * shape3_; \
        (t)->shape1 = shape1_; \
        (t)->shape2 = shape2_; \
        (t)->shape3 = shape3_; \
        (t)->data = (type*) malloc((shape1_) * (shape2_) * (shape3_) * sizeof(*(t)->data)); \
    } while(0)

#define BUFFER_TO_TENSOR(type, buffer, shape1, shape2, shape3) \
    (type) { \
        .data = buffer, \
        .stride1 = shape2 * shape3, \
        .shape1 = shape1, \
        .shape2 = shape2, \
        .shape3 = shape3, \
    }

#define MAT_TO_TENSOR(type, mat) \
    (type) { \
        .data = mat.data, \
        .stride1 = mat.rows * mat.cols, \
        .shape1 = 1, \
        .shape2 = mat.rows, \
        .shape3 = mat.cols, \
    }

#define TENSOR_PRINT(t, shape1, shape2, shape3) \
    do { \
        for(u32 i = 0; i < shape1; ++i) { \
            for(u32 j = 0; j < shape2; ++j) { \
                for(u32 k = 0; k < shape3; ++k) \
                    printf("%u ", *TENSOR3D(t, i, j, k)); \
                printf("\n"); \
            } \
            printf("\n"); \
        } \
    } while(0)

#define DEFINE_TENSOR_INIT(symbol) \
    void tnsr_##symbol##_init(TENSOR_TYPE(symbol)* t, u32 shape1, u32 shape2, u32 shape3);

DEFINE_TENSOR_INIT(u32);
DEFINE_TENSOR_INIT(u16);
DEFINE_TENSOR_INIT(u8);


#endif
