/*
 * ring_buffer.c
 *
 *  Created on: May 24, 2021
 *      Author: benjin
 */

#include "ring_buffer.h"

void init_ring_buffer(struct RingBuffer* rb){
	rb->buffer_ptr = 0;
	rb->last_inference_head = 0;
}

void insert_data(struct RingBuffer* rb, float32_t* data){
	for(int i = 0; i < N_MFCC; i++){
		rb->data[rb->buffer_ptr][i] = data[i];
	}
	increment_buffer_ptr(rb);
}

void increment_buffer_ptr(struct RingBuffer* rb){
	(rb->buffer_ptr)++;
}

void copy_inference_batch(struct RingBuffer* rb, float32_t* batch);

void update_last_inference_head(struct RingBuffer* rb){
	rb->last_inference_head = rb->buffer_ptr;
}
