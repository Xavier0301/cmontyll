#ifndef MAT_H
#define MAT_H

#include "types.h"

#define MAT_TYPE_(symbol) mat_##symbol##_
#define MAT_TYPE(symbol) mat_##symbol

// cols = stride
#define DEFINE_MATRIX_STRUCT(symbol) \
    typedef struct MAT_TYPE_(symbol) { \
        u32 rows; \
        u32 cols; \
        symbol* data; \
    } MAT_TYPE(symbol)

DEFINE_MATRIX_STRUCT(u32);
DEFINE_MATRIX_STRUCT(u16);
DEFINE_MATRIX_STRUCT(u8);

#define DEFINE_MATRIX_STRUCT_WNAME(symbol, name) \
    typedef struct name##_ { \
        u32 rows; \
        u32 cols; \
        symbol* data; \
    } name

#define MAT(t, i, j) ((t).data[((i) * (t).cols) + (j)])
#define MATP(t, i, j) ((t).data + ((i) * (t).cols) + (j))

#define MATRIX_INIT(m, rows, cols, type) \
    do { \
        (m)->rows = rows; \
        (m)->cols = cols; \
        (m)->data = (type*) malloc(rows * cols * sizeof(*(m)->data)); \
    } while(0)

#define MATRIX_PRINT(m, rows, cols) \
    do { \
        for(size_t i = 0; i < rows; ++i) { \
            for(size_t j = 0; j < cols; ++j) \
                printf("%u ", MAT(*m, i, j)); \
            printf("\n"); \
        } \
    } while(0)

#define DEFINE_MATRIX_INIT(symbol) \
    void mat_##symbol##_init(MAT_TYPE(symbol)* m, u32 rows, u32 cols)

DEFINE_MATRIX_INIT(u32);
DEFINE_MATRIX_INIT(u16);
DEFINE_MATRIX_INIT(u8);

u8 mat_u8_min(mat_u8 m);
u8 mat_u8_max(mat_u8 m);
u8 mat_u8_mean(mat_u8 m); 

#endif
