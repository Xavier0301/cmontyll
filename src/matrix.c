#include "matrix.h"

#define INSTANTIATE_MATRIX_INIT(symbol) \
    void mat_##symbol##_init(MAT_TYPE(symbol)* m, u32 rows, u32 cols) { \
        MATRIX_INIT(m, rows, cols, symbol); \
    }

INSTANTIATE_MATRIX_INIT(u32)
INSTANTIATE_MATRIX_INIT(u16)
INSTANTIATE_MATRIX_INIT(u8)

u8 mat_u8_min(mat_u8 m) {
    u8 min = 255;
    for(u32 i = 0; i < m.rows; ++i) {
        for(u32 j = 0; j < m.cols; ++j) {
            if(MAT(m, i, j) < min) min = MAT(m, i, j);
        }
    }
    return min;
}

u8 mat_u8_max(mat_u8 m) {
    u8 max = 0;
    for(u32 i = 0; i < m.rows; ++i) {
        for(u32 j = 0; j < m.cols; ++j) {
            if(MAT(m, i, j) > max) max = MAT(m, i, j);
        }
    }
    return max;
}

u8 mat_u8_mean(mat_u8 m) {
    u32 acc = 0;
    for(u32 i = 0; i < m.rows; ++i) {
        for(u32 j = 0; j < m.cols; ++j) {
            acc += MAT(m, i, j);
        }
    }
    u32 num_elems = m.rows * m.cols;
    return acc / num_elems;
}
