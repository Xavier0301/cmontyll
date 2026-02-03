#ifndef HTM_H
#define HTM_H

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>

#include "math.h"

#include "types.h"
#include "segment.h"
#include "bitarray.h"
#include "lmat.h"

#include "lm_parameters.h"

void htm_prediction_integrate_context(
    u32* output_num_spiking_segments, 
    segment_t** context_pointer, u8** spike_count_pointer, 
    u32* incident_activity, lmat_u32* optional_external_incident_activity,
    enum segment_index_type index_type, u8 num_segments,
    u8 permanence_threshold, u8 segment_spiking_threshold
);

void htm_activate(
    u32* active_columns, 
    u32* activity_bitarrays, u32* prediction_bitarrays, 
    u32 cols, u32 cells
);

void htm_learning_pick_winner_cell(
    u32* winning_cell, u32* winning_spike_count, 
    u8* spike_count_pointer, 
    u32 segments, u32 cells
);

void htm_learning_adjust_permanences(
    u32 cell, 
    segment_t** context_pointer, u8** spike_count_pointer,
    u32* incident_activity, lmat_u32* optional_external_incident_activity,
    u32 cell_is_active, u32 cell_is_predicted,
    u32 col_active_and_unpredicted, u32 winning_cell,
    enum segment_index_type index_type, u8 num_segments,
    htm_params_t htm_p, u32 enable_decay
);

u32 htm_connections_check(
    segment_t* context_pointer, 
    enum segment_index_type index_type, u8 num_segments,
    u32 connections_min, u32 connections_max,
    u32 col_max, u32 cell_max,
    u32 lm_max
);

#endif
