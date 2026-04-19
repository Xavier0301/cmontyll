#include "layer.h"

void layer_init_connections(layer_t* net, layer_params_t p, u32* seed) {
    TENSOR_INIT(&net->out_projections, p.cols, p.cells, p.projections, out_projection_t);

#if PRINT == 2
    u32 size = p.cols * p.cells * p.projections * sizeof(*net->out_projections.data);
    printf("-- layer projection tensor is %u MiB (%u KiB)\n", size >> 20, size >> 10);
#endif

    out_projection_t* out_projections_pointer = net->out_projections.data;
    for(u32 i = 0; i < p.cols * p.cells * p.projections; ++i) {
        u32 out_col = unif_rand_range_u32(0, p.cols, seed);
        u32 out_cell = unif_rand_range_u32(0, p.cells, seed);
        u32 out_seg = unif_rand_range_u32(0, p.segments, seed);

        out_projections_pointer->out_accumulator_pointer = TNSRP(net->segment_accumulators, out_col, out_cell, out_seg);
        out_projections_pointer->permanence = unif_rand_range_u32(0, 255, seed); // random permanence

        out_projections_pointer += 1;
    }
}

void layer_init_state(layer_t* net, layer_params_t p, u32* seed) {
    net->p = p;

    u32 size = p.cols * sizeof(u32);

    net->active = malloc(size);
    net->predicted = malloc(size);
    net->active_prev = malloc(size);

    memset(&net->active, 0, size);
    memset(&net->predicted, 0, size);
    memset(&net->active_prev, 0, size);

#if PRINT == 2
    printf("-- layer state is 3x %u KiB (3x %u B)\n", size >> 10, size);
#endif

    tnsr_u8_init(&net->segment_accumulators, p.cols, p.cells, p.segments);
    size = p.cols * p.cells * p.segments * sizeof(*net->segment_accumulators.data);
    memset(net->segment_accumulators.data, 0, size);

#if PRINT == 2
    printf("-- layer segments accumulator is %u MiB (%u KiB)\n", size >> 20, size >> 10);
#endif
}

void layer_predict(layer_t* net) {
    u8* seg_accumulator_pointer = net->segment_accumulators.data;

    // Segment accumulators -> predicted
    for(u32 col = 0; col < net->p.cols; ++col) {
        u32 pred_bitarray = 0; // 000...000
        for(u32 cell = 0; cell < net->p.cells; ++cell) {
            u32 num_spiking_segs = 0;
            for(u32 seg = 0; seg < net->p.segments; ++seg) {
                num_spiking_segs += (*seg_accumulator_pointer >= net->p.htm.segment_spiking_threshold);
                seg_accumulator_pointer += 1;
            }

            if(num_spiking_segs >= net->p.htm.predicted_threshold) {
                pred_bitarray |= (1U << cell);
            }

        }

        net->predicted[col] = pred_bitarray;
        
    }

    // Reset segment accumulators now that they have been used to generate predictions
    u32 size = net->p.cols * net->p.cells * net->p.segments * sizeof(*net->segment_accumulators.data);
    memset(net->segment_accumulators.data, 0, size);
}

void layer_project(layer_t* net) {
    // active cells -> segment accumulators
    for(u32 col = 0; col < net->p.cols; ++col) {
        for(u32 cell = 0; cell < net->p.cells; ++cell) {
            u32 cell_is_active = GET_BIT(net->active[col], cell);

            // If cell is active, go through its out projections
            if(cell_is_active) {
                out_projection_t* projection_pointer = TNSR_P(net->out_projections, col, cell, 0);
                for(u32 proj = 0; proj < net->p.projections; ++proj) {
                    *(projection_pointer->out_accumulator_pointer) += 1;

                    projection_pointer += 1;
                }
            }
        }
    }
}

void layer_activate(layer_t* net) {

}
 
void layer_learn(layer_t* net) {
    
}
