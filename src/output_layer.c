#include "output_layer.h"

#include "distributions.h"

void init_l3_segment_tensor(output_layer_t* net, feature_layer_params_t f_p, u32* seed) {
    net->in_segments.feedforward = calloc(net->p.cells, sizeof(*net->in_segments.feedforward));
    net->in_segments.internal_context = calloc(net->p.cells * net->p.internal_context_segments, sizeof(*net->in_segments.internal_context));
    net->in_segments.external_context = calloc(net->p.cells * net->p.external_context_segments, sizeof(*net->in_segments.external_context));

    net->spike_count_cache.internal_context_segments = calloc(net->p.cells * net->p.internal_context_segments, sizeof(*net->spike_count_cache.internal_context_segments));
    net->spike_count_cache.external_context_segments = calloc(net->p.cells * net->p.external_context_segments, sizeof(*net->spike_count_cache.external_context_segments));

#if PRINT == 2
    u32 tensor_size = net->p.cells * net->p.internal_context_segments * sizeof(*net->in_segments.internal_context);
    printf("-- output layer internal context segment tensor is %u MiB (%u KiB, %u B)\n", tensor_size >> 20, tensor_size >> 10, tensor_size);

    tensor_size = net->p.cells * net->p.external_context_segments * sizeof(*net->in_segments.external_context);
    printf("-- output layer external context segment tensor is %u MiB (%u KiB, %u B)\n", tensor_size >> 20, tensor_size >> 10, tensor_size);

    tensor_size = net->p.cells * sizeof(*net->in_segments.feedforward);
    printf("-- output layer feedforward segment tensor is %u MiB (%u KiB, %u B)\n", tensor_size >> 20, tensor_size >> 10, tensor_size);

    u32 cache_size = net->p.cells * net->p.internal_context_segments * sizeof(*net->spike_count_cache.internal_context_segments);
    printf(" - output layer internal spike count cache is %u MiB (%u KiB, %u B)\n", cache_size >> 20, cache_size >> 10, cache_size);

    cache_size = net->p.cells * net->p.external_context_segments * sizeof(*net->spike_count_cache.external_context_segments);
    printf(" - output layer external spike count cache is %u MiB (%u KiB, %u B)\n", cache_size >> 20, cache_size >> 10, cache_size);
#endif

    // create feedforward receptive field of l3 cells by connecting them to l4 (feature) cells randomly
    segment_t* feedforward_pointer = net->in_segments.feedforward;

    for(u32 cell = 0; cell < net->p.cells; ++cell) {
        feedforward_pointer->num_connections = unif_rand_range_u32(CONNECTIONS_PER_SEGMENT / 2, CONNECTIONS_PER_SEGMENT, seed); // between 20 and 40 connections

        for(u32 conn = 0; conn < feedforward_pointer->num_connections; ++conn) {
            // connections[conn].index is a union! thus .feature is not a member of a struct, it's just a way to cast to the correct segment type
            feature_index* index_ptr = &(feedforward_pointer->connections[conn].index.feature); 
            index_ptr->col = unif_rand_u32(f_p.cols - 1, seed); // random col index
            index_ptr->cell = unif_rand_u32(f_p.cells - 1, seed); // random cell index

            feedforward_pointer->connections[conn].permanence = unif_rand_u32(255, seed); // random permanence
        }

        feedforward_pointer += 1;
     }


    // create feature context by randomly connecting l3 cells to other l3 cells from the same lm and from other lms ("external l3 cells") 
    segment_t* internal_context_pointer = net->in_segments.internal_context;
    segment_t* external_context_pointer = net->in_segments.external_context;

    u8* internal_context_spike_count_pointer = net->spike_count_cache.internal_context_segments;
    u8* external_context_spike_count_pointer = net->spike_count_cache.external_context_segments;

    for(u32 cell = 0; cell < net->p.cells; ++cell) {
        for(u32 seg = 0; seg < net->p.internal_context_segments; ++seg) {
            *internal_context_spike_count_pointer = 0; // init this cache value to zero
            internal_context_pointer->num_connections = unif_rand_range_u32(CONNECTIONS_PER_SEGMENT / 2, CONNECTIONS_PER_SEGMENT, seed); // between 20 and 40 connections

            for(u32 conn = 0; conn < internal_context_pointer->num_connections; ++conn) {
                // connections[conn].index is a union! thus .feature is not a member of a struct, it's just a way to cast to the correct segment type
                internal_output_index* index_ptr = &(internal_context_pointer->connections[conn].index.internal_output); 
                index_ptr->cell = unif_rand_range_u32(0, net->p.cells - 1, seed); // random cell index
                index_ptr->filler = 0;

                internal_context_pointer->connections[conn].permanence = unif_rand_range_u32(0, 255, seed); // random permanence
            }

            internal_context_pointer += 1;
            internal_context_spike_count_pointer += 1;
        }

        for(u32 seg = 0; seg < net->p.external_context_segments; ++seg) {
            *external_context_spike_count_pointer = 0; // init this cache value to zero
            external_context_pointer->num_connections = unif_rand_range_u32(CONNECTIONS_PER_SEGMENT / 2, CONNECTIONS_PER_SEGMENT, seed); // between 20 and 40 connections

            for(u32 conn = 0; conn < external_context_pointer->num_connections; ++conn) {
                // connections[conn].index is a union! thus .feature is not a member of a struct, it's just a way to cast to the correct segment type
                external_output_index* index_ptr = &(external_context_pointer->connections[conn].index.external_output); 
                index_ptr->cell = unif_rand_range_u32(0, net->p.cells - 1, seed); // random cell index
                index_ptr->lm_id = unif_rand_range_u32(0, net->p.external_lms - 1, seed); // which external lm is chosen as context

                external_context_pointer->connections[conn].permanence = unif_rand_range_u32(0, 255, seed); // random permanence
            }

            external_context_pointer += 1;
            external_context_spike_count_pointer += 1;
        }
    }
}

void init_output_layer(output_layer_t* net, output_layer_params_t p, feature_layer_params_t f_p, u32* seed) {
    net->p = p;

    net->active = calloc(p.cells >> 5, sizeof(*net->active));
    net->active_prev = calloc(p.cells >> 5, sizeof(*net->active_prev));

    net->prediction_scores = calloc(p.cells, sizeof(*net->prediction_scores));

    net->prediction_score_counts = calloc(
        net->p.internal_context_segments + net->p.external_context_segments, 
        sizeof(*net->prediction_score_counts)
    );

    for(u32 i = 0; i < net->p.cells >> 5; ++i) {
        net->active[i] = 0;
        net->active_prev[i] = 0;
    }

    init_l3_segment_tensor(net, f_p, seed);
}

u16 find_kth_largest_from_counts(u16* counts, u32 num_counts, u32 k) {
    u32 elements_seen = 0;
    for (i32 i = num_counts; i >= 0; --i) {
        if(counts[i] > 0) {
            elements_seen += counts[i];
            if (elements_seen >= k) {
                return i; // Found the k-th largest value
            }
        }
    }

    return 0;
}

void output_layer_predict(output_layer_t* net, lmat_u32* external_output_layer_activations) {
    segment_t* internal_context_pointer = net->in_segments.internal_context;
    segment_t* external_context_pointer = net->in_segments.external_context;

    u8* internal_context_spike_count_pointer = net->spike_count_cache.internal_context_segments;
    u8* external_context_spike_count_pointer = net->spike_count_cache.external_context_segments;

    // 1. collect the counts of the cells with top k best context support
    for(u32 cell = 0; cell < net->p.cells; ++cell) {
        u32 num_spiking_segments = 0;

        htm_prediction_integrate_context(&num_spiking_segments, 
            &internal_context_pointer, &internal_context_spike_count_pointer, 
            net->active, NULL,
            INTERNAL_OUTPUT_INDEX_TYPE, net->p.internal_context_segments,
            net->p.extended_htm.context_permanence_threshold, net->p.htm.segment_spiking_threshold
        );

        htm_prediction_integrate_context(&num_spiking_segments, 
            &external_context_pointer, &external_context_spike_count_pointer, 
            NULL, external_output_layer_activations,
            EXTERNAL_OUTPUT_INDEX_TYPE, net->p.external_context_segments,
            net->p.extended_htm.context_permanence_threshold, net->p.htm.segment_spiking_threshold
        );

        if(num_spiking_segments > 255) num_spiking_segments = 255;

        net->prediction_scores[cell] = num_spiking_segments;
    }   
}

u16 find_kth_largest_from_score_counts(u16* score_counts, u32 max_score, u32 k) {
    u32 elements_seen = 0;
    for (i32 i = max_score; i >= 0; --i) {
        if(score_counts[i] > 0) {
            elements_seen += score_counts[i];
            if (elements_seen >= k) {
                return i; // Found the k-th largest value
            }
        }
    }

    return 0;
}

void output_layer_activate(output_layer_t* net, feature_layer_t* f_net, lmat_u32* external_output_layer_activations) {
    // reset frequency counts
    for(u32 i = 0; i < net->p.internal_context_segments + net->p.external_context_segments; ++i) 
        net->prediction_score_counts[i] = 0;

    for(u32 i = 0; i < net->p.cells >> 5; ++i)
        net->active_prev[i] = net->active[i];

    // 1. collect the counts of the cells with top k best context support
    for(u32 cell = 0; cell < net->p.cells; ++cell) {
        u8 score = net->prediction_scores[cell];
        net->prediction_score_counts[score] += 1;
    }    

    // get the exact value of the cell with k-th best context support
    u16 kth_largest = find_kth_largest_from_score_counts(
        net->prediction_score_counts, 
        net->p.internal_context_segments + net->p.external_context_segments - 1, 
        net->p.extended_htm.min_active_cells
    );

    // 2. determine the active cells, those that have enough ffw support 
    //      AND that have more ctx support than the k-th cell with most ctx support (from step 1)

    segment_t* feedforward_pointer = net->in_segments.feedforward;

    for(u32 cell = 0; cell < net->p.cells; ++cell) {
        u32 cell_response = 0;

        for(u32 conn = 0; conn < feedforward_pointer->num_connections; ++conn) {
            segment_data seg_data = feedforward_pointer->connections[conn];

            feature_index index = seg_data.index.feature; 

            u32 cell_is_active = GET_BIT(f_net->active[index.col], index.cell);
            u32 cell_is_connected = seg_data.permanence >= net->p.extended_htm.feedforward_permanence_threshold;

            cell_response += cell_is_active && cell_is_connected;
        }

        if(
            cell_response >= net->p.extended_htm.feedforward_activation_threshold &&
            net->prediction_scores[cell] >= kth_largest
        ) {
            SET_BIT_IN_PACKED32(net->active, cell);
        } else {
            RESET_BIT_IN_PACKED32(net->active, cell);
        }

        feedforward_pointer += 1;
    }

    // 3. Learn context connections       

    segment_t* internal_context_pointer = net->in_segments.internal_context;
    u8* internal_context_spike_count_pointer = net->spike_count_cache.internal_context_segments;

    segment_t* external_context_pointer = net->in_segments.external_context;
    u8* external_context_spike_count_pointer = net->spike_count_cache.external_context_segments;
    
    for(u32 cell = 0; cell < net->p.cells; ++cell) {
        u32 cell_is_active = GET_BIT_FROM_PACKED32(net->active, cell);
        u32 cell_is_predicted = net->prediction_scores[cell] >= 1;
        
        htm_learning_adjust_permanences(cell, 
            &internal_context_pointer, &internal_context_spike_count_pointer,
            net->active_prev, NULL,
            cell_is_active, cell_is_predicted,
            0, 0, // no column, no winning cell
            INTERNAL_OUTPUT_INDEX_TYPE, net->p.internal_context_segments,
            net->p.htm, 0 // no decay
        );

        htm_learning_adjust_permanences(cell, 
            &external_context_pointer, &external_context_spike_count_pointer,
            NULL, external_output_layer_activations,
            cell_is_active, cell_is_predicted,
            0, 0, // no column, no winning cell
            EXTERNAL_OUTPUT_INDEX_TYPE, net->p.external_context_segments,
            net->p.htm, 0 // no decay
        );
    }  

    // TODO: 4. Learn ffw connections (we need a fixed target object id)
    // not doing so is not a huge penalty to how realistic 
    // the current computations are
}

u32 output_layer_get_internal_context_segments_spike_count_cache_bytes(output_layer_params_t p) {
    return p.cells * p.internal_context_segments * sizeof(u8);
}

u32 output_layer_get_external_context_segments_spike_count_cache_bytes(output_layer_params_t p) {
    return p.cells * p.external_context_segments * sizeof(u8);
}

u32 output_layer_get_internal_context_footprint_bytes(output_layer_params_t p) {
    return p.cells * p.internal_context_segments * sizeof(segment_t);
}

u32 output_layer_get_external_context_footprint_bytes(output_layer_params_t p) {
    return p.cells * p.external_context_segments * sizeof(segment_t);
}

u32 output_layer_get_feedforward_footprint_bytes(output_layer_params_t p) {
    return p.cells * sizeof(segment_t);
}

void output_layer_print_memory_footprint(output_layer_params_t p) {
#if PRINT == 2
    u32 tensor_size = output_layer_get_internal_context_footprint_bytes(p);
    printf("-- output layer: internal context segment tensor is %u MiB (%u KiB, %u B)\n", tensor_size >> 20, tensor_size >> 10, tensor_size);
    tensor_size = output_layer_get_external_context_footprint_bytes(p);
    printf("-- output layer: external context segment tensor is %u MiB (%u KiB, %u B)\n", tensor_size >> 20, tensor_size >> 10, tensor_size);

    tensor_size = output_layer_get_feedforward_footprint_bytes(p);
    printf("-- output layer: feedforward segment tensor footprint: %u MiB (%u KiB, %u B)\n", tensor_size >> 20, tensor_size >> 10, tensor_size);

    u32 cache_size = output_layer_get_internal_context_segments_spike_count_cache_bytes(p);
    printf(" - output layer internal segments spike count cache is %u MiB (%u KiB, %u B)\n", cache_size >> 20, cache_size >> 10, cache_size);

    cache_size = output_layer_get_external_context_segments_spike_count_cache_bytes(p);
    printf(" - output layer external segments spike count cache is %u MiB (%u KiB, %u B)\n", cache_size >> 20, cache_size >> 10, cache_size);

    u32 a = (p.cells >> 5) * sizeof(u32);
    u32 ps = p.cells * sizeof(u8);
    u32 psc = (p.internal_context_segments + p.external_context_segments) * sizeof(u16);
    u32 total = a + ps + psc;
    printf("-- output layer state footprint is: %u KiB (%u B)\n", total >> 10, total);
    printf("   \tactive: %u B\n", a);
    printf("   \tactive_prev: %u B\n", a);
    printf("   \tprediction_scores: %u KiB (%u B)\n", ps >> 10, ps);
    printf("   \tprediction_score_counts: %u B\n", psc);
#else
    (void) p;
#endif
}
