//
// Created by Janus Bo Andersen, Dec 2020
//

#ifndef KNN_DATASET_H
#define KNN_DATASET_H

#include <vector>
#include <string>
#include <cstdint>

#define REF_SIZE 1024
#define FEATURE_DIM 63
#define QUERY_SIZE 100

namespace mms {
    const uint8_t reference_vectors[REF_SIZE * FEATURE_DIM] = {
        #include "data/ref_vals.txt"
    };

    const uint8_t reference_labels[REF_SIZE] = {
        #include "data/ref_labels.txt"
    };

    const uint8_t query_vectors[QUERY_SIZE * FEATURE_DIM] = {
        #include "data/query_vals.txt"
    };

    const uint8_t query_labels[REF_SIZE] = {
        #include "data/query_labels.txt"
    };

    const std::vector<uint8_t> class_ids {1, 2, 3, 4, 5};
    const std::vector<std::string> class_names {"blue", "yellow", "brown", "red", "green"};

}
#endif //KNN_DATASET_H
