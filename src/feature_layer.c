#include "feature_layer.h"

#include "distributions.h"

void init_l4_segment_tensor(feature_layer_t* net, location_layer_params_t location_layer_p) {
    // t->cols = cols; t->cells = cells; t->segments = segments;

    net->in_segments.context = calloc(net->p.cols * net->p.cells * net->p.segments, sizeof(*net->in_segments.context));

    u32 tensor_size = net->p.cols * net->p.cells * net->p.segments * sizeof(*net->in_segments.context);
    printf("-- feature layer segment tensor is %u MiB (%u KiB, %u B)\n", tensor_size >> 20, tensor_size >> 10, tensor_size);

    // create feature context by randomly connecting l4 cells together, and connection l6 cells to l4 cells
    segment_t* segments_pointer = net->in_segments.context;

    for(u32 col = 0; col < net->p.cols; ++col) {
        for(u32 cell = 0; cell < net->p.cells; ++cell) {
            for(u32 seg = 0; seg < net->p.feature_segments; ++seg) {
                segments_pointer->connection_count = 0; // init this cache value to zero
                segments_pointer->num_connections = unif_rand_range_u32(CONNECTIONS_PER_SEGMENT / 2, CONNECTIONS_PER_SEGMENT); // between 20 and 40 connections

                for(u32 conn = 0; conn < segments_pointer->num_connections; ++conn) {
                    // connections[conn].index is a union! thus .feature is not a member of a struct, it's just a way to cast to the correct segment type
                    feature_index* index_ptr = &(segments_pointer->connections[conn].index.feature); 
                    index_ptr->col = unif_rand_u32(net->p.cols - 1); // random column index
                    index_ptr->cell = unif_rand_u32(net->p.cells - 1); // random cell index

                    segments_pointer->connections[conn].permanence = unif_rand_u32(255); // random permanence
                }

                segments_pointer += 1;
            }

            for(u32 seg = net->p.feature_segments; seg < net->p.segments; ++seg) {
                segments_pointer->connection_count = 0; // init this cache value to zero
                segments_pointer->num_connections = unif_rand_range_u32(CONNECTIONS_PER_SEGMENT / 2, CONNECTIONS_PER_SEGMENT); // between 20 and 40 connections

                for(u32 conn = 0; conn < segments_pointer->num_connections; ++conn) {
                    // connections[conn].index is a union! thus .feature is not a member of a struct, it's just a way to cast to the correct segment type
                    location_index* index_ptr = &(segments_pointer->connections[conn].index.location); 
                    index_ptr->module = unif_rand_u32(location_layer_p.modules - 1); // random module
                    index_ptr->cell_x = unif_rand_u32(location_layer_p.module_params[index_ptr->module].cells_sqrt - 1); // random cell
                    index_ptr->cell_y = unif_rand_u32(location_layer_p.module_params[index_ptr->module].cells_sqrt - 1); // random cell

                    segments_pointer->connections[conn].permanence = unif_rand_u32(255); // random permanence
                }

                segments_pointer += 1;
            }
        }        
    }
}

void init_feature_layer(feature_layer_t* net, feature_layer_params_t p, location_layer_params_t location_layer_p) {
    // assert(net->p.segments == net->p.feature_segments + net->p.location_segments, )
    net->p = p;

    net->active = calloc(p.cols, sizeof(*net->active));
    net->predicted = calloc(p.cols, sizeof(*net->predicted));

    net->active_prev = calloc(p.cols, sizeof(*net->active_prev));

    for(u32 col = 0; col < net->p.cols; ++col) {
        net->active[col] = 0;
        net->predicted[col] = 0;

        net->active_prev[col] = 0;
    }

    init_l4_segment_tensor(net, location_layer_p);
}

/**
 * @brief 
 * 
 * @param net 
 * @param location_activity is a bitarray where each entry represents a cell that has a bitarray of which module has that cell active
 */
void feature_layer_predict(feature_layer_t* net, location_layer_t* location_net) {
    segment_t* segments_pointer = net->in_segments.context; // start at the beginning

    for(u32 col = 0; col < net->p.cols; ++col) {
        u32 pred_bitarray = 0; // 000...000
        for(u32 cell = 0; cell < net->p.cells; ++cell) {
            u32 cell_is_predicted = 0;
            for(u32 seg = 0; seg < net->p.segments; ++seg) {
                segments_pointer->connection_count = 0; // reset this cache

                u32 cell_accumulator = 0;
                for(u32 conn = 0; conn < segments_pointer->num_connections; ++conn) {
                    segment_data seg_data = segments_pointer->connections[conn];

                    u32 is_cell_active;
                    if(seg < net->p.feature_segments) {
                        // seg_data.index is a union! thus .feature is not a member of a struct, it's just a way to cast to the correct segment type
                        feature_index index = seg_data.index.feature;
                        is_cell_active = GET_BIT(net->active[index.col], index.cell);
                    } else {
                        // seg_data.index is a union! thus .feature is not a member of a struct, it's just a way to cast to the correct segment type
                        location_index index = seg_data.index.location;
                        is_cell_active = location_layer_check_active(location_net, index);
                    }

                    u32 is_cell_connected = seg_data.permanence >= net->p.permanence_threshold;

                    cell_accumulator += is_cell_active & is_cell_connected;
                }

                if(cell_accumulator > 255) cell_accumulator = 255;
                segments_pointer->connection_count = cell_accumulator;

                if(cell_accumulator > net->p.segment_spiking_threshold) cell_is_predicted = 1;

                segments_pointer += 1;
            }

            pred_bitarray |= (cell_is_predicted << cell);
        }
        
        net->predicted[col] = pred_bitarray;
    }
}

/**
 * @brief Active cells are the cells that are 
 * 
 * @param net 
 * @param active_columns of shape (net->p.cols), caller initialized
 */
void feature_layer_activate(feature_layer_t* net, u8* active_columns, location_layer_t* location_net) {
    // predicted -> active

    for(u32 col_it = 0; col_it < net->p.cols; ++col_it) {
        net->active_prev[col_it] = net->active[col_it];

        if(active_columns[col_it]) {
            u16 active_col = col_it;

            u32 act_bitarray = 0; // 000...000 
            for(u32 cell = 0; cell < net->p.cells; ++cell) {
                u32 was_predicted = GET_BIT(net->predicted[active_col], cell);
                act_bitarray |= (was_predicted << cell);
            }
            // if no cell was activated through predictions, activate all cells in column
            if(act_bitarray == 0) act_bitarray = ~ ((u32) 0); // 111...111

            net->active[active_col] = act_bitarray;
        } else {
            net->active[col_it] = 0; // 000...0000
        }

    }

    // learning (involves traversing the segments data)

    segment_t* segments_pointer = net->in_segments.context; // start at the beginning

    for(u32 col = 0; col < net->p.cols; ++col) {
        u32 pred_bitarray = net->predicted[col];

        /**
         * If the columns is actived but no cell was predicted, we have to select the "winning cell"
         * to that end, we iterate through all the cells and we find the cell with the segment that was closest to becoming active
         */
        u32 winning_cell = 0;
        u32 winning_segment_connections = 0;
        if(active_columns[col] && pred_bitarray == 0) {
            // We find the cell with the segment that had most activity
            segment_t* seg_ptr_copy = segments_pointer;
            for(u32 cell = 0; cell < net->p.cells; ++cell) {
                for(u32 seg = 0; seg < net->p.segments; ++seg) {
                    if(seg_ptr_copy->connection_count > winning_segment_connections) {
                        winning_cell = cell;
                        winning_segment_connections = seg_ptr_copy->connection_count ;
                    }

                    seg_ptr_copy += 1;
                }
            }
        }

        for(u32 cell = 0; cell < net->p.cells; ++cell) {
            u32 cell_is_predicted = GET_BIT(pred_bitarray, cell);

            for(u32 seg = 0; seg < net->p.segments; ++seg) {
                /** There are two cases for a reinforcement:
                 * 
                 * 1. If the column is active, the cell is predicted and the segment was spiking,
                 *      we select that segment for reinforcement. That means that we increase the permanences
                 *      of active incident cells and decrease the permanences of inactive incident cells on the segment
                 * 
                 * 2. If the column is active, no cell in the col is predicted and this cell is the winning cell (chosen prior)
                 *      we 
                 */
                u32 seg_was_spiking = segments_pointer->connection_count >= net->p.segment_spiking_threshold;

                u32 should_reinforce_case1 = active_columns[col] && cell_is_predicted 
                    && seg_was_spiking;
                u32 should_reinforce_case2 = active_columns[col] && pred_bitarray == 0 
                    && cell == winning_cell 
                    && segments_pointer->connection_count == winning_segment_connections;

                u32 should_reinforce = should_reinforce_case1 && should_reinforce_case2;
                
                if(should_reinforce) {
                    for(u32 conn = 0; conn < segments_pointer->num_connections; ++conn) {
                        segment_data* seg_data = &(segments_pointer->connections[conn]);

                        u32 incident_cell_was_active;
                        if(seg < net->p.feature_segments) {
                            feature_index index = seg_data->index.feature;

                            incident_cell_was_active = GET_BIT(
                                net->active_prev[index.col], 
                                index.cell
                            );
                        } else {
                            location_index index = seg_data->index.location;

                            incident_cell_was_active = location_layer_check_active_prev(location_net, index);
                        }
                        
                        // we don't care if the cell was connected (perm > thresh), we just reward active and
                        // punish inactive for a spiking segment
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
                
                // If the cell was predicted but ended up not become active, apply a decay to
                // synapses above perm thresh and connected to an active cell
                u32 should_decay = !active_columns[col] && cell_is_predicted && seg_was_spiking;
                if(should_decay) {
                    for(u32 conn = 0; conn < segments_pointer->num_connections; ++conn) {
                        segment_data* seg_data = &(segments_pointer->connections[conn]);

                        u32 incident_cell_was_active;
                        if(seg < net->p.feature_segments) {
                            feature_index index = seg_data->index.feature;

                            incident_cell_was_active = GET_BIT(
                                net->active_prev[index.col], 
                                index.cell
                            );
                        } else {
                            location_index index = seg_data->index.location;

                            incident_cell_was_active = location_layer_check_active_prev(location_net, index);
                        }

                        if(incident_cell_was_active && seg_data->permanence >= net->p.permanence_threshold) {
                            seg_data->permanence = safe_sub_u8(
                                seg_data->permanence, 
                                net->p.perm_decay
                            );
                        }
                    }
                }

                segments_pointer += 1;
            }
        }  
    }
}
