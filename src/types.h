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

typedef struct vec2d_ {
    i32 x;
    i32 y;
} vec2d;

typedef struct uvec2d_ {
    u32 x;
    u32 y;
} uvec2d;

inline static int is_vec2d_positive(vec2d v) {
    return v.x >= 0 && v.y >= 0;
}

static inline u8 safe_add_u8(u8 a, u8 b) {
    if(255 - a < b) return 255;
    else return a + b;
}

static inline u8 safe_sub_u8(u8 a, u8 b) {
    if(a < b) return 0;
    else return a - b;
}

#endif // TYPES_H
