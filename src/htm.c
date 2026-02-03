#include "htm.h"

void htm_prediction_integrate_context(
    u32* output_num_spiking_segments, 
    segment_t** context_pointer, u8** spike_count_pointer, 
    u32* incident_activity, lmat_u32* optional_external_incident_activity,
    enum segment_index_type index_type, u8 num_segments,
    u8 permanence_threshold, u8 segment_spiking_threshold
) {
    for(u32 seg = 0; seg < num_segments; ++seg) {
        
        u32 cell_accumulator = 0;
        for(u32 conn = 0; conn < (*context_pointer)->num_connections; ++conn) {
            segment_data seg_data = (*context_pointer)->connections[conn];

            u32 is_cell_active;
            if(index_type == INTERNAL_OUTPUT_INDEX_TYPE) {
                internal_output_index index = seg_data.index.internal_output;
                is_cell_active = GET_BIT_FROM_PACKED32(incident_activity, index.cell);
            } else if(index_type == EXTERNAL_OUTPUT_INDEX_TYPE) {
                external_output_index index = seg_data.index.external_output;
                u32* lm_output = LMATP(*optional_external_incident_activity, index.lm_id, 0);
                is_cell_active = GET_BIT_FROM_PACKED32(
                    lm_output, 
                    index.cell
                );
            } else if(index_type == FEATURE_INDEX_TYPE) {
                feature_index index = seg_data.index.feature;
                is_cell_active = GET_BIT(incident_activity[index.col], index.cell);
            } else /* if(index_type == LOCATION_INDEX_TYPE) */ {
                location_index index = seg_data.index.location;
                is_cell_active = GET_BIT(incident_activity[index.col], index.cell);
            }

            u32 is_cell_connected = seg_data.permanence >= permanence_threshold;

            cell_accumulator += is_cell_active & is_cell_connected;
        }

        if(cell_accumulator > 255) cell_accumulator = 255;
        **spike_count_pointer = cell_accumulator;

        if(cell_accumulator > segment_spiking_threshold) 
            *output_num_spiking_segments += 1;

        *context_pointer += 1;
        *spike_count_pointer += 1;
    }
}

void htm_activate(
    u32* active_columns, 
    u32* activity_bitarrays, u32* prediction_bitarrays, 
    u32 cols, u32 cells
) {
    for(u32 col = 0; col < cols; ++col) {
        if(active_columns[col] != 0) {
            u32 act_bitarray = 0; // 000...000 
            for(u32 cell = 0; cell < cells; ++cell) {
                u32 was_predicted = GET_BIT(prediction_bitarrays[col], cell);
                act_bitarray |= (was_predicted << cell);
            }
            // if no cell was activated through predictions, activate all cells in column
            if(act_bitarray == 0) act_bitarray = ~ ((u32) 0); // 111...111

            activity_bitarrays[col] = act_bitarray;
        } else {
            activity_bitarrays[col] = 0; // 000...0000
        }
    } 
}

void htm_learning_pick_winner_cell(
    u32* winning_cell, u32* winning_spike_count, 
    u8* spike_count_pointer, 
    u32 segments, u32 cells
) {
    for(u32 cell = 0; cell < cells; ++cell) {
        for(u32 seg = 0; seg < segments; ++seg) {
            if(*spike_count_pointer > *winning_spike_count) {
                *winning_cell = cell;
                *winning_spike_count = *spike_count_pointer;
            }

            spike_count_pointer += 1;
        }
    }
}

void htm_learning_adjust_permanences(
    u32 cell, 
    segment_t** context_pointer, u8** spike_count_pointer,
    u32* incident_activity, lmat_u32* optional_external_incident_activity,
    u32 cell_is_active, u32 cell_is_predicted,
    u32 col_active_and_unpredicted, u32 winning_cell,
    enum segment_index_type index_type, u8 num_segments,
    htm_params_t htm_p, u32 enable_decay
) { 
    u32 cell_will_not_reinforce = (!cell_is_active || !cell_is_predicted) 
        && (!col_active_and_unpredicted || cell != winning_cell);
    u32 cell_will_not_decay = !enable_decay || !cell_is_predicted || cell_is_active;

    if(cell_will_not_reinforce && cell_will_not_decay) {
        *context_pointer += num_segments;
        *spike_count_pointer += num_segments;

        return;
    }

    for(u32 seg = 0; seg < num_segments; ++seg) {
        /** There are two cases for a reinforcement:
         * 
         * 1. If the column is active, the cell is predicted and the segment was spiking,
         *      we select that segment for reinforcement. That means that we increase the permanences
         *      of active incident cells and decrease the permanences of inactive incident cells on the segment
         * 
         * 2. If the column is active, no cell in the col is predicted and this cell is the winning cell (chosen prior)
         *      we 
         */
        u32 seg_was_spiking = **spike_count_pointer >= htm_p.segment_spiking_threshold;

        u32 should_reinforce_case1 = cell_is_active 
            && cell_is_predicted 
            && seg_was_spiking;
        u32 should_reinforce_case2 = col_active_and_unpredicted
            && cell == winning_cell;

        u32 should_reinforce = should_reinforce_case1 || should_reinforce_case2;
        
        if(should_reinforce) {
            for(u32 conn = 0; conn < (*context_pointer)->num_connections; ++conn) {
                segment_data* seg_data = &((*context_pointer)->connections[conn]);

                // if we don't know how to handle the incident cell index, we default to not handling anything
                u32 incident_cell_was_active = 0; 
                if(index_type == FEATURE_INDEX_TYPE) {
                    feature_index index = seg_data->index.feature;
                    incident_cell_was_active = GET_BIT(incident_activity[index.col], index.cell);
                } else if(index_type == LOCATION_INDEX_TYPE) {
                    location_index index = seg_data->index.location;
                    incident_cell_was_active = GET_BIT(incident_activity[index.col], index.cell);
                } else if(index_type == INTERNAL_OUTPUT_INDEX_TYPE) {
                    internal_output_index index = seg_data->index.internal_output;
                    incident_cell_was_active = GET_BIT_FROM_PACKED32(incident_activity, index.cell);
                } else if(index_type == EXTERNAL_OUTPUT_INDEX_TYPE) {
                    external_output_index index = seg_data->index.external_output;
                    u32* lm_output = LMATP(*optional_external_incident_activity, index.lm_id, 0);
                    incident_cell_was_active = GET_BIT_FROM_PACKED32(
                        lm_output, 
                        index.cell
                    );
                }
                
                // we don't care if the cell was connected (perm > thresh), we just reward active and
                // punish inactive for a spiking segment
                if(incident_cell_was_active) {
                    seg_data->permanence = safe_add_u8(
                        seg_data->permanence, 
                        htm_p.perm_increment
                    );
                } else {
                    seg_data->permanence = safe_sub_u8(
                        seg_data->permanence, 
                        htm_p.perm_decrement
                    );
                }
            }
        } 

        // If the cell was predicted but ended up not become active, apply a decay to
        // synapses above perm thresh and connected to an active cell
        u32 should_decay = enable_decay && cell_is_predicted && !cell_is_active && seg_was_spiking;
        if(should_decay) {
            for(u32 conn = 0; conn < (*context_pointer)->num_connections; ++conn) {
                segment_data* seg_data = &((*context_pointer)->connections[conn]);

                // if we don't know how to handle the incident cell index, we default to not handling anything
                u32 incident_cell_was_active = 0; 
                if(index_type == FEATURE_INDEX_TYPE) {
                    feature_index index = seg_data->index.feature;
                    incident_cell_was_active = GET_BIT(incident_activity[index.col], index.cell);
                } else if(index_type == LOCATION_INDEX_TYPE) {
                    location_index index = seg_data->index.location;
                    incident_cell_was_active = GET_BIT(incident_activity[index.col], index.cell);
                }

                if(incident_cell_was_active && seg_data->permanence >= htm_p.permanence_threshold) {
                    seg_data->permanence = safe_sub_u8(
                        seg_data->permanence, 
                        htm_p.perm_decay
                    );
                }
            }
        }

        *context_pointer += 1;
        *spike_count_pointer += 1;
    }
}

u32 htm_connections_check(
    segment_t* context_pointer, 
    enum segment_index_type index_type, u8 num_segments,
    u32 connections_min, u32 connections_max,
    u32 col_max, u32 cell_max,
    u32 lm_max
) {
    for(u32 seg = 0; seg < num_segments; ++seg) {

        if(context_pointer->num_connections < connections_min 
            || context_pointer->num_connections > connections_max
        ) { return 0; }

        for(u32 conn = 0; conn < (context_pointer)->num_connections; ++conn) {
            segment_data seg_data = (context_pointer)->connections[conn];

            if(index_type == FEATURE_INDEX_TYPE) {
                feature_index index = seg_data.index.feature;

                if(index.col > col_max) return 0;
                if(index.cell > cell_max) return 0;
            } else if(index_type == LOCATION_INDEX_TYPE) {
                location_index index = seg_data.index.location;

                if(index.col > col_max) return 0;
                if(index.cell > cell_max) return 0;
            } else if(index_type == INTERNAL_OUTPUT_INDEX_TYPE) {
                internal_output_index index = seg_data.index.internal_output;

                if(index.cell > cell_max) return 0;
            } else if(index_type == EXTERNAL_OUTPUT_INDEX_TYPE) {
                external_output_index index = seg_data.index.external_output;

                if(index.cell > cell_max) return 0;
                if(index.lm_id > lm_max) return 0;
            }
            
        }
        context_pointer += 1;
    }

    return 1;
}
