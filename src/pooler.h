#ifndef POOLER_H
#define POOLER_H

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>

#include "math.h"

#include "types.h"
#include "lookup_table.h"
#include "tensor.h"
#include "distributions.h"

typedef struct pooler_params_t_ {
    u32 learning_enabled;
    u32 boosting_enabled;

    u32 num_inputs;
    u32 num_minicols;

    // from the pooling paper: "Performance of the SP is not sensitive to the connection threshold parameter."
    u8 permanence_threshold; // threshold for a permanence value to mean 'connected' (i.e. 128 cuts of half of inputs)
    u8 stimulus_threshold; // threshold for actually activating a column
    u8 activation_density; // target percent of column activated. 2% of 255 is ~5.10 => hence 0.02 is coded as 5 in u8

    u16 top_k; // the number of columns that will actually get to be active (only the most active do get active)

    u8 permanence_increment; // amount by which we should increase synaptice permanence (if input bit on)
    u8 permanence_decrement; // amount by which we should decrease synaptice permanence (if input bit off)

    u32 log2_activation_window; // log2 of the window of activations for boosting 
                                // This means that they are constrained to be powers of 2, for computational efficiency

    f32 boosting_strength; // important for calculating boosting factors, used for col responses and calculated at the end of the cycle
} pooler_params_t;

typedef struct pooler_t_ {
    pooler_params_t params;

    mat_u8 synaptic_permanences; // For no topology. Of shape (#minicols, #inputs) => extend to tensor4d for topology

    u8* column_responses; // For no topology. Of shape (#minicols).
    u8* column_responses_copy; // Copy to be used in quickselect. For no topology. Of shape (#minicols).
    u8* column_activations; // For no topology. Of shape (#minicols).

    u16* time_averaged_activations; // Of shape (#minicols)

    u8* boosting_factors; // of shape (#minicols).

    lookup_table_i8 boosting_LUT; // of length (boosting_LUT_length). Represents the function x -> e^(-beta(x - alpha)) in range [0; boosting_LUT_length - 1]
        // 1. If x <= alpha then it represents that actual boosting factor that you can readily multiply with the reponse
        // 2. If x > alpha then it comes in a negative number which absolute value represents the amount to be shifted left (because our arch does not support 8-bit divs)
} pooler_t;

void init_pooler(pooler_t* p, u32 num_inputs, u32 num_minicols, f32 p_connected, u32 learning_enabled, u32 boosting_enabled);

void print_pooler(pooler_t* p);

// A spatial pooler is a set of mini columns
//      - Each mini column is connected to a subset of the input space
//      - We will assume that each minicolumn has a potential connection to the whole input space for now

void pooler_step(pooler_t* p, u8* input, u32 num_inputs);

#endif // POOLER_H
