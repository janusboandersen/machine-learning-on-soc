#ifndef KNN_H_
#define KNN_H_

#include <stdint.h>

#define DATA_DIM 63		// features in a feature vector
#define DATA_SIZE 1024	// size of the reference (training) set


// copy will infer DMA (FastDMA) transfer to PL (from contiguous mem area!), and specify copy sizes
// data_access is strictly required to be sequential for speed-up
#pragma SDS data copy(ref_vecs[0:DATA_SIZE*DATA_DIM], query[0:DATA_DIM], d2[0:DATA_SIZE])
#pragma SDS data access_pattern(ref_vecs:SEQUENTIAL, query:SEQUENTIAL, d2:SEQUENTIAL)
#pragma SDS data mem_attribute(ref_vecs:PHYSICAL_CONTIGUOUS, query:PHYSICAL_CONTIGUOUS, d2:PHYSICAL_CONTIGUOUS)
void knn_hw(uint8_t* ref_vecs, uint8_t* query, unsigned int* d2);

#endif
