#include "pooler.h"

#include "assertf.h"
#include "algorithms.h"

/**
 * @brief there are two connectivity layers to this pooling stuff:
 *  1. The input-output connectivity:
 *      Here each column has an assigned input region (regions can intersect)
 *  2. The output-output connectivity:
 *      Here the columns can inhibit other columns in their region
 *      Only the top reacting columns are selected
 */

// For now, there is global input connectivity. If topology is on, change that shit.
void init_connections(pooler_t* p, u32 num_inputs, u32 num_minicols, f32 p_connected, u32* seed) {
    // to how many inputs could each minicolumn be connected?
    u32 num_inputs_connected = (u32) (p_connected * (f32) num_inputs);

    matrix_u8_init(&p->synaptic_permanences, num_minicols, num_inputs);

    for(u32 col = 0; col < num_minicols; ++col) {
        for(u32 input = 0; input < num_inputs; ++input) {
            u8 potential_connection = (u8) unif_rand_range_u32(0, 255, seed);
            if(potential_connection >= num_inputs_connected) 
                MAT(p->synaptic_permanences, col, input) = (u8) unif_rand_range_u32(0, 255, seed);
            else
                MAT(p->synaptic_permanences, col, input) = 0;
        }
    }
}

void init_columns(pooler_t* p, u32 num_minicols) {
    p->column_responses = (u8*) calloc(num_minicols, sizeof(*p->column_responses));
    p->column_responses_copy = (u8*) calloc(num_minicols, sizeof(*p->column_responses_copy));
    p->column_activations = (u8*) calloc(num_minicols, sizeof(*p->column_activations));

    if(p->params.boosting_enabled) {
        p->time_averaged_activations = (u16*) calloc(num_minicols, sizeof(*p->time_averaged_activations));
        p->boosting_factors = (u8*) calloc(num_minicols, sizeof(*p->time_averaged_activations));

        u16 initial_averaged_acts = (u16) (p->params.activation_density << 8); // loss of precision here but small one
        for(u32 minicol = 0; minicol < num_minicols; ++minicol) p->time_averaged_activations[minicol] = initial_averaged_acts;
        for(u32 minicol = 0; minicol < num_minicols; ++minicol) p->boosting_factors[minicol] = 1;
    }

    // The following is optional
    // for(u32 col = 0; col < num_minicols; ++col) p->column_activations[col] = 0;
    // for(u32 col = 0; col < num_minicols; ++col) p->column_responses[col] = 0;
    // for(u32 col = 0; col < num_minicols; ++col) p->column_responses_copy[col] = 0;
}

// function is f: x -> e^(-beta (x - s)) for x in the range [0;1] and s in the range [0;1]
// If we have x' and s' be in the range [0;255] and be of form x' = 255 * x and s' = 255 * s
//      Then the function if of form f: x -> e^(-beta/255 (x - s))
void init_boosting_LUT(pooler_t* p) {
    f32 beta = p->params.boosting_strength / 255.0;
    f32 s = p->params.activation_density; 

    // Let's calculate the range of inputs for which this function is > 1/128. Lower values we don't care about.
    f32 thresh = 1.0/128.0;
    u32 length = 0;
    for(u8 x = 0; x <= 255 && length == 0; ++x) {
        f32 res = exp(- beta * (x - s));

        if(res <= thresh) {
            length = x;
            break;
        }
    }
    if(length == 0) length = 255;

    // Knowing the number of values we will use, we can finally instantiate our LUT
    lut_i8_init(&p->boosting_LUT, 0, length);

    // We re-run the computation and this time we store it in our LUT
    for(u8 x = 0; x < length; ++x) {
        f32 res = exp(- beta * (x - s));

        if(res > 0.5) p->boosting_LUT.data[x] = (i8) round(res); // if the res is > 0.5, we can just apply the rounded boosting factor as is with an 8-bit multiplier
        else p->boosting_LUT.data[x] = (i8) round(log2(res)); // for values <= 0.5 we wanna divide but dividing is like shifting: num / a = num >> log2(a)
    }
}

// expects numbers in [0;1]
#define REPR_u8(x) ((u8) (round(x * 255.0f)))

void init_params(pooler_t* p, u32 num_inputs, u32 num_minicols, u32 learning_enabled, u32 boosting_enabled) {
    p->params.learning_enabled = learning_enabled;
    p->params.boosting_enabled = boosting_enabled;

    p->params.num_inputs = num_inputs;
    p->params.num_minicols = num_minicols;

    p->params.permanence_threshold = REPR_u8(0.5); // theta_c
    p->params.stimulus_threshold = 1; // theta_stim. threshold for actually activating a column

    f32 activation_density = 0.02; // s (2% sparsity)
    p->params.activation_density = REPR_u8(activation_density); // target percent of column activated. 2% of 255 is ~5.10 => hence 0.02 is coded as 5 in u8
    
    p->params.top_k = (u16) round(activation_density * num_minicols); // k. determines the number of columns to activate to reach desired sparsity

    p->params.permanence_increment = REPR_u8(0.1f); // p+
    p->params.permanence_decrement = REPR_u8(0.02f); // p-

    p->params.log2_activation_window = (u32) log2(1024); // T 

    p->params.boosting_strength = 100.0f; // beta
}

void init_pooler(pooler_t* p, u32 num_inputs, u32 num_minicols, f32 p_connected, u32 learning_enabled, u32 boosting_enabled, u32* seed) {
    init_params(p, num_inputs, num_minicols, learning_enabled, boosting_enabled);

    init_connections(p, num_inputs, num_minicols, p_connected, seed);
    init_columns(p, num_minicols);
    if(p->params.boosting_enabled) init_boosting_LUT(p);
    else p->boosting_LUT.length = 0;
}

void print_pooler(pooler_t* p) {
#if PRINT == 2
    printf("=== pooler ===\n");
    printf("params\n");
    printf("\tlearning = %u  boosting = %u\n", 
        p->params.learning_enabled, p->params.boosting_enabled);
    printf("\tnum_inputs = %u  num_minicols = %u\n", 
        p->params.num_inputs, p->params.num_minicols);
    printf("\tpermanence_thresh [θ_c] = %u (%lf)\n", 
        p->params.permanence_threshold, p->params.permanence_threshold / 255.0);
    printf("\tstimulus_thresh [θ_stim] = %u\n", 
        p->params.stimulus_threshold);
    printf("\tactivation_density [s] = %u (%lf) top_k [k] = %u\n", 
        p->params.activation_density, p->params.activation_density / 255.0, p->params.top_k);
    printf("\tpermanence_increment [p+] = %u (%lf)  permanence_decrement [p-] = %u (%lf)\n", 
        p->params.permanence_increment, p->params.permanence_increment / 255.0, p->params.permanence_decrement, p->params.permanence_decrement / 255.0);
    printf("\tlog2_activation_window [T] = %u (%u)\n", 
        p->params.log2_activation_window, 1 << p->params.log2_activation_window);
    if(p->params.boosting_enabled) {
        printf("\tboosting_strength [beta] = %f\n", 
            p->params.boosting_strength);

        printf("boosting LUT (length = %u, default_value = %u)\n", 
            p->boosting_LUT.length, p->boosting_LUT.default_value);
        for(u32 x = 0; x < p->boosting_LUT.length; ++x)
            printf("%u -> %d, ", x, p->boosting_LUT.data[x]);
    }
    printf("\n");
#else
    (void) p;
#endif
}

u16 calculate_next_time_averaged_acts(u16 last_time_avgs_acts, u32 col_is_activated, u32 log2_activation_window) {
    // We have the formula aa_i(t) = ((T-1)*aa_i(t-1) + a_i(t))/T.
    // If we let X = aa_i(t-1) and Y = a_i(t) then we have to compute:
    //      ((T-1)X + Y)/T = (TX - X + Y)/T = X + (Y-X)/T
    // If T is a power of 2 then we have to compute
    //      X - ((X-Y) >> log(T))
    // That's for f: [0;1] -> [0;1]
    // But we want g: [0; B] -> [0; B] where B could be 2^16-1 or 2^32-1
    // Thus we need to transform the input and the output:
    //      g: X -> f(x/B) * B = X + (YB - X) / T

    if(!col_is_activated)
        return last_time_avgs_acts - (last_time_avgs_acts >> log2_activation_window);

    u16 B = 65535; // B=2^16 - 1 => encoding is on u16 i.e. on [0; 2^16 - 1]
    u16 diff = B - last_time_avgs_acts;
    return last_time_avgs_acts + (diff >> log2_activation_window);
}

/**
 * @brief 
 * 
 * @param input 
 * @param num_inputs 
 */
void pooler_step(pooler_t* p, u8* input, u32 num_inputs) {
    u32 num_minicols = p->synaptic_permanences.rows;
    u32 num_inputs_ = p->synaptic_permanences.cols;

    assertf(num_inputs == num_inputs_, "pooling input has unexpected shape (got: %u | expec: %u)", num_inputs, num_inputs_);

    u8* perm_pointer = p->synaptic_permanences.data;
    for(u32 minicol = 0; minicol < num_minicols; ++minicol) {
        u32 feedforward_accumulator = 0;
        for(u32 input_it = 0; input_it < num_inputs; ++input_it) {
            u8 perm = *perm_pointer;
            if(perm >= p->params.permanence_threshold) 
                feedforward_accumulator += input[input_it]; 

            perm_pointer += 1;
        }
        if(feedforward_accumulator > 255) feedforward_accumulator = 255; // clamp to [0; 255] to cast to u8 after => basically impossible to go over 255 tho
        u8 feedforward_accumulator_u8 = (u8) feedforward_accumulator;

        u8 col_response = 0;

        if(p->params.boosting_enabled) {
            // apply boosting factor to col_resp
            i8 boosting_factor = p->boosting_factors[minicol];

            if(boosting_factor < 0) {
                col_response = feedforward_accumulator_u8 >> (-boosting_factor);
            } else {
                u32 mul_res = feedforward_accumulator_u8 * (u8) boosting_factor; // 8x8 bit multiplication in a 16-bits result extended to 32-bits

                if(mul_res > 255) col_response = 255; // clamp to [0; 255] to cast to u8 after => basically impossible to go over 255 tho
                else col_response = (u8) mul_res;
            }
        } else {
            col_response = feedforward_accumulator_u8;
        }

        p->column_responses[minicol] = col_response;
        p->column_responses_copy[minicol] = col_response;
    }
    // How to efficiently compute whether or not a column is in the top 2%?
    //      (1) Quickselect the column activation at the 98 percentile 
    //      (2) Filter elements that are greater

    u32 top_k_value = quickselect(p->column_responses_copy, 
        0, num_minicols - 1, 
        num_minicols - p->params.top_k /* because we want k-LARGEST not k-smallest*/
    );

    // printf("column_responses (top_k_value=%u):\n", top_k_value);
    // for(u32 it = 0; it < num_minicols; ++it) {
    //     printf("%u ", p->column_responses[it] >= top_k_value);
    // }
    // printf("\n");

    perm_pointer = p->synaptic_permanences.data; // reset perm_pointer to beginning of permanence data block

    u32 selected_cols_counter = 0; // The number of columns selected so far
        // Important since
    // (1) calculating column activations, then (2) apply hebbian learning rules and (3) calculate boosting factors
    for(u32 minicol = 0; minicol < num_minicols; ++minicol) {
        // (1) calculating column activations
        u32 response = p->column_responses[minicol];
        u32 col_is_activated = (response >= top_k_value) 
            && (response >= p->params.stimulus_threshold)
            && (selected_cols_counter <= p->params.top_k);

        selected_cols_counter += col_is_activated;

        p->column_activations[minicol] = col_is_activated;

        // (2) Apply hebbian learning rules: learning the synapses connected to this minicolumn (learning happens only for activated columns!)
        if(p->params.learning_enabled && col_is_activated) {
            for(u32 input_it = 0; input_it < num_inputs; ++input_it) {
                i32 perm = (i32) *perm_pointer;

                if(input[input_it] == 1) {
                    // reward the connection to this input bit
                    perm += p->params.permanence_increment;
                    if(perm > 255) perm = 255;
                } else {
                    // punish the connection to this input bit
                    perm -= p->params.permanence_decrement;
                    if(perm < 0) perm = 0;
                }

                *perm_pointer = (u8) perm;
                
                perm_pointer += 1;
            }
        }

        if(p->params.boosting_enabled) {
            // (3) calculate boosting factors by (3.1) calculating the time-averaged activations of this column and then finally (3.2) calculating the boosting factors 
            //      (3.1)Updating time-averaged activations for this minicolumn
            u16 next_time_averaged_acts = calculate_next_time_averaged_acts(
                p->time_averaged_activations[minicol], 
                col_is_activated, 
                p->params.log2_activation_window);
            p->time_averaged_activations[minicol] = next_time_averaged_acts;

            //      (3.2)Calculate the boosting factor for this minicolumn
            u8 next_time_averaged_acts_u8 = (u8) (next_time_averaged_acts >> 8); // drops 8-bits of precision in the repr
            p->boosting_factors[minicol] = lut_i8_lookup(&p->boosting_LUT, next_time_averaged_acts_u8);
        }
    }
}
