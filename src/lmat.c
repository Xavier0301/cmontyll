#include "lmat.h"

#define INSTANTIATE_LMAT_INIT(symbol) \
    void lmat_##symbol##_init(LMAT_TYPE(symbol)* m, u16 rows, u8 log_cols) { \
        LMAT_INIT(m, rows, log_cols, symbol); \
    }

INSTANTIATE_LMAT_INIT(u32)
INSTANTIATE_LMAT_INIT(u16)
INSTANTIATE_LMAT_INIT(u8)

#define INSTANTIATE_LMAT_COUNT(symbol) \
    u32 lmat_##symbol##_count(LMAT_TYPE(symbol)* m) { \
        return LMAT_COUNT(m); \
    }

INSTANTIATE_LMAT_COUNT(u32)
INSTANTIATE_LMAT_COUNT(u16)
INSTANTIATE_LMAT_COUNT(u8)
