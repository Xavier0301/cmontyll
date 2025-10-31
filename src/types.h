#ifndef TYPES_H
#define TYPES_H

#include <stdint.h>

#define REPR_u8(x) ((u8) (round(x * 255.0f)))

typedef int8_t i8;
typedef int16_t i16;
typedef int32_t i32;
typedef int64_t i64;

typedef uint8_t u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;

typedef float f32;
typedef double f64;

#define PAIR_TYPE_(symbol) pair_##symbol##_
#define PAIR_TYPE(symbol) pair_##symbol

#define DEFINE_PAIR_STRUCT(symbol) \
    typedef struct PAIR_TYPE_(symbol) { \
        symbol first; \
        symbol second; \
    } PAIR_TYPE(symbol)

DEFINE_PAIR_STRUCT(u8);
DEFINE_PAIR_STRUCT(u16);
DEFINE_PAIR_STRUCT(u32);

#define INDEX2D_TYPE_(symbol) index2d_##symbol##_
#define INDEX2D_TYPE(symbol) index2d_##symbol

#define DEFINE_INDEX2D_STRUCT(symbol) \
    typedef struct INDEX2D_TYPE_(symbol) { \
        symbol x; \
        symbol y; \
    } INDEX2D_TYPE(symbol)

DEFINE_INDEX2D_STRUCT(u8);
DEFINE_INDEX2D_STRUCT(u16);
DEFINE_INDEX2D_STRUCT(u32);

static inline u8 safe_add_u8(u8 a, u8 b) {
    return (255 - a < b) ? 255 : a + b;
}

static inline u8 safe_sub_u8(u8 a, u8 b) {
    return (a < b) ? 0 : a - b;
}

#endif // TYPES_H
