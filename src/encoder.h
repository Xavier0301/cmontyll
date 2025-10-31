#ifndef ENCODER_H
#define ENCODER_H

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>

#include "math.h"

#include "types.h"
#include "sparse.h"

void encode_integer(u8* output, u32 output_length, u32 non_null_count, u32 input, u32 min, u32 max);

#endif // ENCODER_H
