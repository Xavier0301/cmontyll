#ifndef BITARRAY_H
#define BITARRAY_H

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>

#include "math.h"

#include "types.h"

#define BIT_MASK(i) (1 << i)

#define GET_BIT(x, i) ((x & BIT_MASK(i)) >> i)

// --- Packed bitarray

/**
 * @brief For a packed array of form u32* elements
 * we gotta leave 32 = 2^5 bits of adressing for the bitarray 
 * Meaning if u32 index is our address, we do:
 *      bitarray = elements[index >> 5] 
 *      bit = GET_BIT(bitarray, index & 0b11111)
 * 
 * To set a bit, it is:
 *      elements[index >> 5] |= (1 << (index & 0b11111))
 * 
 * To reset a bit, it is:
 *      elements[index >> 5] &= ~ (1 << (index & 0b11111))
 */

#define GET_BIT_FROM_PACKED32(packed, index) (GET_BIT(packed[index >> 5], index & 0b11111))
#define SET_BIT_IN_PACKED32(packed, index) (packed[index >> 5] |= (1 << (index & 0b11111)))
#define RESET_BIT_IN_PACKED32(packed, index) (packed[index >> 5] &= ~ (1 << (index & 0b11111)))


#define GET_BIT_FROM_PACKED8(packed, index) (GET_BIT(packed[index >> 3], index & 0b111))
#define SET_BIT_IN_PACKED8(packed, index) (packed[index >> 3] |= (1 << (index & 0b111)))
#define RESET_BIT_IN_PACKED8(packed, index) (packed[index >> 3] &= ~ (1 << (index & 0b111)))

#endif
