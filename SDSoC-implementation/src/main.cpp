/*
 * Advanced Digital Design (E5ADD) 2020
 * Implementation of k-Nearest Neighbours on Zynq-7000
 * M&Ms colour classification system
 * Oct-Dec 2020
 *
 * Files:
 * knn.h			HW accelerator prototype and SDS interfaces
 * knn.cpp			HW accelerator implementation
 * knn_support.h	SW KNN algorithm, voting function and scoring
 * dataset.h		Contains metadata and loads the flat files in data/
 * ref_vals.txt		1024 reference vectors (training) with 63 features
 * ref_labels.txt	Labels for each of the 1024 reference vectors
 * query_vals.txt	100 query vectors (test) with 63 features
 * query_labels.txt	Labels for each of the 100 query vectors
 *
 */

#include <knn.h>
#include <iostream>
#include <cstring>
#include <stdlib.h>
#include <climits>
#include <stdio.h>
#include "sds_utils.h"
#include <vector>
#include <algorithm>
#include <stdint.h>
#include "knn_support.h"
#include "dataset.h"


int main(int argc, char** argv)
{

	// Get K from user
	int knn_k;
	std::cout << "Perform k-Nearest Neighbours with k = ?  Please enter a positive integer:" << std::endl;
	std::cin >> knn_k;
	std::cout << "Starting KNN with k=" << knn_k << "." << std::endl;

    // Heap allocation
    uint8_t* ref_vecs   = (uint8_t*) sds_alloc(sizeof(uint8_t) * DATA_SIZE * DATA_DIM);  // Reference (train) set
    uint8_t* query      = (uint8_t*) sds_alloc(sizeof(uint8_t) * QUERY_SIZE * DATA_DIM); // Query (test) set
    unsigned int* d2_hw = (unsigned int*) sds_alloc(sizeof(unsigned int)*DATA_SIZE);     // 1024 return vals

    // Heap allocation for PS algo.
    unsigned int* d2_sw = (unsigned int*) malloc(sizeof(unsigned int)*DATA_SIZE);

    // Ensure OK memory alloc.
    if((ref_vecs == NULL) || (query == NULL) || (d2_hw == NULL) || (d2_sw == NULL)){
       std::cout << "Could not allocate on the heap." << std::endl;
       return -1;
     }

    // Load datasets onto heap to ensure similar access from both HW and SW
    for (int i = 0; i < DATA_SIZE*DATA_DIM; i++) {
    	ref_vecs[i] = mms::reference_vectors[i];
    }

    for (int i = 0; i < QUERY_SIZE*DATA_DIM; i++) {
        query[i] = mms::query_vectors[i];
    }


    // Use the profiling method from SDSoC guide
    sds_utils::perf_counter hw_ctr, sw_ctr;


    // Classify all 100 query vectors using the hardware accelerator
	std::vector<int> hw_results;
	uint8_t* q_ptr = query;

	// profile HW algo. only during HW runtime
	for (int q = 0; q < QUERY_SIZE*DATA_DIM; q += DATA_DIM) {

		q_ptr = query + q;	// advance pointer to next query vector
		hw_ctr.start();

		// Compute KNN with hardware accelerator for q'th query vector
		#pragma SDS data mem_attribute(ref_vecs:PHYSICAL_CONTIGUOUS, q_ptr:PHYSICAL_CONTIGUOUS, d2_hw:PHYSICAL_CONTIGUOUS)
		knn_hw(ref_vecs, q_ptr, d2_hw);
		//knn::knn_hw(mms::reference_vectors, mms::query_vectors + q, d2, DATA_SIZE, DATA_DIM);
		hw_ctr.stop();

		// Classify this vector based on squared distances in d2
		auto cls_idx = knn::find_nearest_neighbours(knn_k, d2_hw, mms::reference_labels, DATA_SIZE, mms::class_ids);
		hw_results.push_back(cls_idx); // Store, these are array indices, not the classes themselves
	}

	// Compare results and rate
	std::cout << "*** Hardware KNN results ***" << std::endl;
	auto hw_accuracy = knn::knn_rate(hw_results, mms::query_labels, QUERY_SIZE, mms::class_ids);
	std::cout << "Accuracy: " << hw_accuracy << std::endl;


    // Classify all 100 query vectors using the software implementation on Arm Cortex-A9
	std::vector<int> sw_results;

	// profile SW algo. only during equivalent runtime to the HW
	for (int q = 0; q < QUERY_SIZE*DATA_DIM; q += DATA_DIM) {

		q_ptr = query + q;	// advance pointer to next query vector
		sw_ctr.start();

		// Compute KNN in software for q'th query vector
		knn::knn_sw(ref_vecs, q_ptr, d2_sw, DATA_SIZE, DATA_DIM);

		sw_ctr.stop();

		// Classify this vector based on squared distances in d2
		auto cls_idx = knn::find_nearest_neighbours(knn_k, d2_sw, mms::reference_labels, DATA_SIZE, mms::class_ids);
		sw_results.push_back(cls_idx); // Store, these are array indices, not the classes themselves
	}

	// Compare results and rate
	std::cout << "*** Software KNN results ***" << std::endl;
	auto sw_accuracy = knn::knn_rate(sw_results, mms::query_labels, QUERY_SIZE, mms::class_ids);
	std::cout << "Accuracy: " << sw_accuracy << std::endl;


	// Verify HW algorithm: Confirm that HW and SW get same results, sqdiff must be zero
	unsigned long sq_diff = 0;
	for(int i = 0; i < DATA_SIZE; i++) {
		sq_diff += SQ(d2_sw[i] - d2_hw[i]);
	}

	std::cout << "*** Verification report ***" << std::endl;
	if(sq_diff != 0) {
		std::cout << "BAD: HW and SW Euclidian distances do not match." << std::endl;
	} else {
		std::cout << "OK: HW and SW Euclidian distances match." << std::endl;
	}


	std::cout << "*** Performance report ***" << std::endl;
    uint64_t sw_cycles = sw_ctr.avg_cpu_cycles();
    uint64_t hw_cycles = hw_ctr.avg_cpu_cycles();

    double speedup = (double) sw_cycles / (double) hw_cycles;

    std::cout << "Avg. CPU cycles running KNN in software: " << sw_cycles << std::endl;
    std::cout << "Avg. CPU cycles running KNN in hardware: " << hw_cycles << std::endl;
    std::cout << "Speed-up: " << speedup << std::endl;


    // Free heap
    sds_free(ref_vecs);
    sds_free(query);
    sds_free(d2_hw);
    free(d2_sw);

    return 0;
}
