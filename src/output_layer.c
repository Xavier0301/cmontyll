#include "output_layer.h"

#include "distributions.h"

void init_l3_segment_tensor(output_layer_t* net, feature_layer_params_t feature_layer_p) {
    net->in_segments.feedforward = calloc(net->p.cells, sizeof(*net->in_segments.feedforward));
    net->in_segments.context = calloc(net->p.cells * net->p.context_segments, sizeof(*net->in_segments.context));

    u32 tensor_size = net->p.cells * net->p.context_segments * sizeof(*net->in_segments.context);
    printf("-- output layer context segment tensor is %u MiB (%u KiB, %u B)\n", tensor_size >> 20, tensor_size >> 10, tensor_size);

    tensor_size = net->p.cells * sizeof(*net->in_segments.feedforward);
    printf("-- output layer feedforward segment tensor is %u MiB (%u KiB, %u B)\n", tensor_size >> 20, tensor_size >> 10, tensor_size);

    // create feedforward receptive field of l3 cells by connecting them to l4 (feature) cells randomly
    segment_t* ffw_segments_ptr = net->in_segments.context;

    for(u32 cell = 0; cell < net->p.cells; ++cell) {
        
        ffw_segments_ptr->connection_count = 0; // init this cache value to zero
        ffw_segments_ptr->num_connections = unif_rand_range_u32(CONNECTIONS_PER_SEGMENT / 2, CONNECTIONS_PER_SEGMENT); // between 20 and 40 connections

        for(u32 conn = 0; conn < ffw_segments_ptr->num_connections; ++conn) {
            // connections[conn].index is a union! thus .feature is not a member of a struct, it's just a way to cast to the correct segment type
            feature_index* index_ptr = &(ffw_segments_ptr->connections[conn].index.feature); 
            index_ptr->col = unif_rand_u32(feature_layer_p.cols - 1); // random col index
            index_ptr->cell = unif_rand_u32(feature_layer_p.cells - 1); // random cell index

            ffw_segments_ptr->connections[conn].permanence = unif_rand_u32(255); // random permanence
        }

        ffw_segments_ptr += 1;
     }


    // create feature context by randomly connecting l3 cells to other l3 cells from the same lm and from other lms ("external l3 cells") 
    segment_t* context_segments_pointer = net->in_segments.context;

    for(u32 cell = 0; cell < net->p.cells; ++cell) {
        for(u32 seg = 0; seg < net->p.internal_context_segments; ++seg) {
            context_segments_pointer->connection_count = 0; // init this cache value to zero
            context_segments_pointer->num_connections = unif_rand_range_u32(CONNECTIONS_PER_SEGMENT / 2, CONNECTIONS_PER_SEGMENT); // between 20 and 40 connections

            for(u32 conn = 0; conn < context_segments_pointer->num_connections; ++conn) {
                // connections[conn].index is a union! thus .feature is not a member of a struct, it's just a way to cast to the correct segment type
                internal_output_index* index_ptr = &(context_segments_pointer->connections[conn].index.internal_output); 
                index_ptr->cell = unif_rand_u32(net->p.cells - 1); // random cell index

                context_segments_pointer->connections[conn].permanence = unif_rand_u32(255); // random permanence
            }

            context_segments_pointer += 1;
        }

        // for(u32 seg = net->p.internal_context_segments; seg < net->p.context_segments; ++seg) {
        //     segments_pointer->was_spiking = 0; // init this cache value to zero
        //     segments_pointer->num_connections = unif_rand_range_u32(CONNECTIONS_PER_SEGMENT / 2, CONNECTIONS_PER_SEGMENT); // between 20 and 40 connections

        //     for(u32 conn = 0; conn < segments_pointer->num_connections; ++conn) {
        //         // connections[conn].index is a union! thus .feature is not a member of a struct, it's just a way to cast to the correct segment type
        //         external_output_index* index_ptr = &(segments_pointer->connections[conn].index.external_output); 
        //         index_ptr->module = unif_rand_u32(location_layer_p.modules - 1); // random module
        //         index_ptr->cell_x = unif_rand_u32(location_layer_p.module_params[index_ptr->module].cells_sqrt - 1); // random cell
        //         index_ptr->cell_y = unif_rand_u32(location_layer_p.module_params[index_ptr->module].cells_sqrt - 1); // random cell

        //         segments_pointer->connections[conn].permanence = unif_rand_u32(255); // random permanence
        //     }

        //     segments_pointer += 1;
        // }
    }
}

void init_output_layer(output_layer_t* net, output_layer_params_t p, feature_layer_params_t feature_layer_p) {
    net->p = p;

    net->active = calloc(p.cells >> 5, sizeof(*net->active));
    net->active_prev = calloc(p.cells >> 5, sizeof(*net->active_prev));

    net->prediction_scores = calloc(p.cells, sizeof(*net->prediction_scores));

    net->prediction_score_counts = calloc(p.context_segments, sizeof(*net->prediction_score_counts));

    for(u32 i = 0; i < net->p.cells >> 5; ++i) {
        net->active[i] = 0;
        net->active_prev[i] = 0;
    }

    init_l3_segment_tensor(net, feature_layer_p);
}

u16 find_kth_largest_from_counts(u16* counts, u32 num_counts, u32 k) {
    u32 elements_seen = 0;
    for (u32 i = num_counts; i >= 0; --i) {
        if(counts[i] > 0) {
            elements_seen += counts[i];
            if (elements_seen >= k) {
                return i; // Found the k-th largest value
            }
        }
    }

    return 0;
}

void output_layer_predict(output_layer_t* net) {
    segment_t* ctx_segments_pointer = net->in_segments.context;
    // 1. collect the counts of the cells with top k best context support
    for(u32 cell = 0; cell < net->p.cells; ++cell) {
        u32 active_ctx_segments = 0;
        for(u32 seg = 0; seg < net->p.internal_context_segments; ++seg) {
            u32 seg_accumulator = 0;
            for(u32 conn = 0; conn < ctx_segments_pointer->num_connections; ++conn) {
                segment_data seg_data = ctx_segments_pointer->connections[conn];
                internal_output_index index = seg_data.index.internal_output;

                u32 cell_is_active = GET_BIT_FROM_PACKED32(net->active, index.cell);
                u32 cell_is_connected = seg_data.permanence >= net->p.context_permanence_threshold;

                seg_accumulator += cell_is_active && cell_is_connected;
            }

            u32 segment_is_spiking = seg_accumulator >= net->p.segment_spiking_threshold;
            if(seg_accumulator > 255) seg_accumulator = 255;
            ctx_segments_pointer->connection_count = seg_accumulator;

            active_ctx_segments += segment_is_spiking;


            ctx_segments_pointer += 1;
        }

        if(active_ctx_segments > 255) active_ctx_segments = 255;

        net->prediction_scores[cell] = active_ctx_segments;
    }   
}

void output_layer_activate(output_layer_t* net, feature_layer_t* feature_net) {
    // reset frequency counts
    for(u32 i = 0; i < net->p.context_segments; ++i) 
        net->prediction_score_counts[i] = 0;

    for(u32 i = 0; i < net->p.cells >> 5; ++i)
        net->active_prev[i] = net->active[i];

    // 1. collect the counts of the cells with top k best context support
    for(u32 cell = 0; cell < net->p.cells; ++cell) {
        u8 score = net->prediction_scores[cell];
        net->prediction_score_counts[score] += 1;
    }    

    // get the exact value of the cell with k-th best context support
    u16 kth_largest = find_kth_largest_from_counts(
        net->prediction_score_counts, 
        net->p.context_segments, net->p.min_active_cells
    );

    // 2. determine the active cells, those that have enough ffw support 
    //      AND that have more ctx support than the k-th cell with most ctx support (from step 1)

    segment_t* ffw_segments_pointer = net->in_segments.feedforward;

    for(u32 cell = 0; cell < net->p.cells; ++cell) {
        u32 cell_response = 0;

        for(u32 conn = 0; conn < ffw_segments_pointer->num_connections; ++conn) {
            segment_data seg_data = ffw_segments_pointer->connections[conn];

            // connections[conn].index is a union! thus .feature is not a member of a struct, it's just a way to cast to the correct segment type
            feature_index index = seg_data.index.feature; 

            u32 cell_is_active = GET_BIT(feature_net->active[index.col], index.cell);
            u32 cell_is_connected = seg_data.permanence >= net->p.feedforward_permanence_threshold;

            cell_response += cell_is_active && cell_is_connected;
        }

        if(
            cell_response >= net->p.feedforward_activation_threshold &&
            net->prediction_scores[cell] >= kth_largest
        ) {
            SET_BIT_IN_PACKED32(net->active, cell);
        } else {
            RESET_BIT_IN_PACKED32(net->active, cell);
        }

        ffw_segments_pointer += 1;
    }

    // TODO 3. Learn context connections       

    segment_t* ctx_segments_pointer = net->in_segments.context;
    
    for(u32 cell = 0; cell < net->p.cells; ++cell) {
        u32 cell_is_predicted = net->prediction_scores[cell] >= 1;
        u32 cell_is_active = GET_BIT_FROM_PACKED32(net->active, cell);

        u32 predicted_and_active = cell_is_predicted && cell_is_active;

        if(!predicted_and_active) {
            // go to the next ieration directly
            ctx_segments_pointer += net->p.context_segments;
        } else {
            for(u32 seg = 0; seg < net->p.internal_context_segments; ++seg) {
                u32 seg_was_spiking = ctx_segments_pointer->connection_count >= net->p.segment_spiking_threshold;

                u32 should_reinforce = cell_is_active && cell_is_predicted && seg_was_spiking;

                if(should_reinforce) {
                    for(u32 conn = 0; conn < ctx_segments_pointer->num_connections; ++conn) {
                        segment_data* seg_data = &(ctx_segments_pointer->connections[conn]);
                        
                        internal_output_index index = seg_data->index.internal_output;

                        u32 incident_cell_was_active = GET_BIT_FROM_PACKED32(net->active, index.cell);

                        if(incident_cell_was_active) {
                            seg_data->permanence = safe_add_u8(
                                seg_data->permanence, 
                                net->p.perm_increment
                            );
                        } else {
                            seg_data->permanence = safe_sub_u8(
                                seg_data->permanence, 
                                net->p.perm_decrement
                            );
                        }
                    }
                }

                // omitting decay

                ctx_segments_pointer += 1;
            }
        }
    }  

    // TODO: 4. Learn ffw connections (we need a fixed target object id)
}
