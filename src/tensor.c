#include "tensor.h"

#define INSTANTIATE_TENSOR_INIT(symbol) \
    void tnsr_##symbol##_init(TENSOR_TYPE(symbol)* t, u32 shape1, u32 shape2, u32 shape3) { \
        TENSOR_INIT(t, shape1, shape2, shape3, symbol); \
    }

INSTANTIATE_TENSOR_INIT(u16)
INSTANTIATE_TENSOR_INIT(u8)
