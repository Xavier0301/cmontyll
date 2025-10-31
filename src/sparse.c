#include "sparse.h"

#include "stdio.h"

void init_spvec_u1(spvec_u1* v, u16 length, u16 non_null_count) {
    v->length = length;
    v->non_null_count = non_null_count;
    v->indices = calloc(non_null_count, sizeof(*v->indices));
}

void print_spvec_u8(u8* v, u32 length) {
    u32 ones = 0;
    for(u32 i = 0; i < length; ++i) 
        ones += (v[i] == 1);

    printf("[%u, %u, %f] ", length, ones, ((f32) ones) / (f32) length);

    // print the whole thing if it's small enough
    if(length <= 100) {
        for(u32 i = 0; i < length; ++i) 
            printf("%u ", v[i]);
        printf("\n");
    }
}

#include "bitarray.h"

void print_packed_spvec_u32(u32* v, u32 unpacked_length) {
    u32 ones = 0;
    for(u32 i = 0; i < unpacked_length; ++i) {
        ones += (GET_BIT_FROM_PACKED32(v, i) == 1);
    }

    printf("[%u, %u, %f] ", unpacked_length, ones, ((f32) ones) / (f32) unpacked_length);

    // print the whole thing if it's small enough
    if(unpacked_length <= 100) {
        for(u32 i = 0; i < unpacked_length; ++i) 
            printf("%u ", GET_BIT_FROM_PACKED32(v, i));
        printf("\n");
    }
}
