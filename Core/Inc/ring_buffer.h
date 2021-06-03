/*
 * ring_buffer.h
 *
 *  Created on: May 24, 2021
 *      Author: benjin
 */

#ifndef INC_RING_BUFFER_H_
#define INC_RING_BUFFER_H_

#define BUFFERSIZE 93
#define N_MFCC 13
#define MIN_DIST 20 // Minimal distance between buffer_ptr and last_inference_head before doing inference again

#include <arm_math.h>

struct RingBuffer {
	int8_t data[BUFFERSIZE][N_MFCC];
	uint16_t last_inference_head;
	uint16_t buffer_ptr;
	bool filled;
	bool triggered;
};

void set_triggered(struct RingBuffer* rb);

void init_ring_buffer(struct RingBuffer* rb);

void insert_data(struct RingBuffer* rb, int8_t* data);

void increment_buffer_ptr(struct RingBuffer* rb);

void copy_inference_batch(struct RingBuffer* rb, int8_t* batch);

void update_last_inference_head(struct RingBuffer* rb);

int distance_buffer_ptr_last_inference_head(struct RingBuffer* rb);

bool do_inference(struct RingBuffer* rb);


#endif /* INC_RING_BUFFER_H_ */
