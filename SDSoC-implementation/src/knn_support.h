/********************************************
 * Finding K Nearest Neighbours Algorithm
 * Created by Janus Bo Andersen, Dec 2020
 *
 * find_nearest_neighbours:
 * In array of squared distances, "d2"
 * with corresponding labels in "labels",
 * both of length "size",
 * the algo finds "K" smallest distances,
 * and by counting votes for each class
 * based on the classes from "all_classes",
 * the most frequent (mode) is picked.
 * if it is a tie, the class for the nearest
 * neighbour is chosen.
 *
 * knn_sw: Compute sq. Euclidian distances
 *
 * knn_rate: Rate the accuracy of the classifier
 *
 *********************************************/

#ifndef KNN_KNN_SUPPORT_H
#define KNN_KNN_SUPPORT_H
#define SQ(x) ((x)*(x))

#include <vector>
#include <algorithm>
#include <cstdint>

namespace knn {
    int find_nearest_neighbours(const int K,
                                const unsigned int* distances,
                                const uint8_t* labels,
                                const int size,
                                const std::vector<uint8_t>& all_classes
    ) {

        // Store distance-label pairs, we sort the first and get label from the second
        std::vector<std::pair<int, uint8_t>> dl_pairs;
        for (int i = 0; i < size; i++) {
            dl_pairs.push_back(std::make_pair(distances[i], labels[i]));
        }

        // Sort the pairs in place from lowest to highest
        std::sort(dl_pairs.begin(), dl_pairs.end());

        // For the K lowest distances, extract the class label
        std::vector<int> k_classes;
        for (int k = 0; k < K; k++) {
            k_classes.push_back(dl_pairs[k].second);    // grab the class label
        }

        // Count votes for each class in the original class list
        std::vector<int> k_count;
        for(auto& cls: all_classes) {
            k_count.push_back(std::count(k_classes.begin(), k_classes.end(), cls));
        }

        // Find the most frequent and translate it into a class index
        auto k_maxvote_itr = std::max_element(k_count.begin(), k_count.end());
        auto k_maxvote_idx = std::distance(k_count.begin(), k_maxvote_itr);

        // Returns the index for the most popular class in all_classes
        return (int) k_maxvote_idx;
    }

    void knn_sw(const uint8_t* ref_vecs,  // "size" reference vectors each of length "dim"
                const uint8_t* query,     // 1 query vector of length "dim"
                unsigned int* d2,         // allocated space to store sq. Euclid. dists.
                int size,                 // number of reference vectors
                int dim                   // number of features in a vector
                )
    {

        unsigned long sum_dist;

        // Nested loop to go through all reference vectors
        for (int i = 0; i < size; i++) {
            sum_dist = 0;

            // MAC operation to go through all vector components
            for (int j = 0; j < dim; j++) {
                sum_dist += SQ(query[j] - ref_vecs[i*dim + j]);
            }

            // Store the squared Euclidian distance
            d2[i] = sum_dist;
        }
    }

    float knn_rate(const std::vector<int>& results,      // array indices
                   const uint8_t* query_labels,
                   const int query_size,
                   const std::vector<uint8_t>& class_ids

                   ) {

        // Compare results and rate
        int predict;
        int truth;
        int missed = 0;
        for (int q = 0; q < query_size; q++) {
            predict = class_ids[results[q]];
            truth = (int) query_labels[q];

            if (predict != truth) {
                missed++;
                std::cout << "Misclassification of obs. " << q+1 << " out of " << query_size << "." <<
                          " Pred.: " << predict << ", True: " << truth << std::endl;
            }
        }

        return 1 - missed/(float) query_size;

    }

}


#endif //KNN_KNN_SUPPORT_H
