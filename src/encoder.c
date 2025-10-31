#include "encoder.h"

/**
 * @brief 
 * 
 * @param output need to be initialized by the caller!
 * @param input 
 * @param range 
 * @param num_bits 
 * @param num_active_bits 
 */
void encode_integer(u8* output, u32 output_length, u32 non_null_count, u32 input, u32 min, u32 max) {
    u32 num_buckets = output_length - non_null_count + 1;

    u32 i = (num_buckets * (input - min)) / (max - min);

    // Let's calculate the output on the fly 
    // We could cache these results or pre-compute them so that we only need 
    //      to calculate i and then copy the cached result to the output array
    // Transforms a bunch of logic and arithmetic into 1 copy operation, nothing major saved
    // HOWEVER, if we can just pass around the POINTER to the sdr, then it's pretty major
    //      since it transforms a O(num_bits) into a O(1)
    // However this requires that the output stay unchanged in the later stages
    //      which I believe it is!

    for(u32 j = 0; j < i; ++j) output[j] = 0;
    for(u32 j = i; j < i + non_null_count; ++j) output[j] = 1;
    for(u32 j = i + non_null_count; j < output_length; ++j) output[j] = 0;
}
