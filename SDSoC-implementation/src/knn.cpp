
#include "knn.h"

#define SQ(x) ((x)*(x))

void knn_hw(
			uint8_t *ref_vecs,  // Reference vectors (training data)
			uint8_t *query,     // Query vector (test vector)
			unsigned int *d2    // Sq. Euclidian distances return value
		  )
{

	// BRAM block for local memory
    uint8_t query_local[DATA_DIM];

    // Partition the local BRAM for more concurrency. A high factor puts data in FFs.
	#pragma HLS ARRAY_PARTITION variable=query_local cyclic factor=32 dim=1

    // Running sum
    unsigned int sum_dist = 0;

    // Get entire query vector (from DMA) to local BRAM to be read multiple times
    get_query: for(int i = 0; i < DATA_DIM; i++){
    #pragma HLS PIPELINE
        query_local[i] = query[i];
    }

    // Perfect nested loop
    for_each_ref_vec: for(int i = 0; i < DATA_SIZE; i++) {
        compute_dist: for(int j = 0; j < DATA_DIM; j++) {
        #pragma HLS PIPELINE
		#pragma HLS UNROLL factor=32

            // MAC operation to compute sq. Euclidian distance, getting ref_vecs sequentially
            sum_dist += SQ(query_local[j] - ref_vecs[i*DATA_DIM + j]);

            if(j == DATA_DIM-1) {
            	d2[i] = sum_dist;  	// send final sq. dist. to i-th ref.vec.
            	sum_dist = 0; 		// reset
            }

        }
    }
}
