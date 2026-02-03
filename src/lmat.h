#ifndef LMAT_H
#define LMAT_H

#include <stdint.h>
#include <stdlib.h>

#include "types.h"

#define LMAT_TYPE_(symbol) lmat_##symbol##_
#define LMAT_TYPE(symbol) lmat_##symbol

// cols = stride
#define DEFINE_LMAT_STRUCT(symbol) \
    typedef struct LMAT_TYPE_(symbol) { \
        symbol* data; \
        u16 rows; \
        u8 log_cols; \
    } LMAT_TYPE(symbol)

DEFINE_LMAT_STRUCT(u32);
DEFINE_LMAT_STRUCT(u16);
DEFINE_LMAT_STRUCT(u8);

#define LMAT(t, i, j) ((t).data[((i) << (t).log_cols) + (j)])
#define LMATP(t, i, j) ((t).data + ((i) << (t).log_cols) + (j))

#define LMAT_INIT(m, rows, log_cols, type) \
    do { \
        (m)->rows = rows; \
        (m)->log_cols = log_cols; \
        (m)->data = (type*) calloc((rows) << (log_cols), sizeof(*(m)->data)); \
    } while(0)

#define LMAT_PRINT(m, rows, cols) \
    do { \
        for(size_t i = 0; i < rows; ++i) { \
            for(size_t j = 0; j < cols; ++j) \
                printf("%u ", MAT(m, i, j)); \
            printf("\n"); \
        } \
    } while(0)

#define LMAT_COUNT(m) ((m)->rows << (m)->log_cols)

#define DEFINE_LMAT_INIT(symbol) \
    void lmat_##symbol##_init(LMAT_TYPE(symbol)* m, u16 rows, u8 log_cols)

DEFINE_LMAT_INIT(u32);
DEFINE_LMAT_INIT(u16);
DEFINE_LMAT_INIT(u8);

#define DEFINE_LMAT_COUNT(symbol) \
    u32 lmat_##symbol##_count(LMAT_TYPE(symbol)* m)

DEFINE_LMAT_COUNT(u32);
DEFINE_LMAT_COUNT(u16);
DEFINE_LMAT_COUNT(u8);

#endif
