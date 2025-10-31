#include "interfaces.h"

void init_features(features_t* f, u32 length, u32 non_null_bits) {
    f->active_columns = calloc(length, sizeof(*f->active_columns));

    f->num_columns = length;
    f->num_active_columns = non_null_bits;
}
