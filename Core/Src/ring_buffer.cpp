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

void insert_data(struct RingBuffer* rb, int8_t* data){
	for(int i = 0; i < N_MFCC; i++){
		rb->data[rb->buffer_ptr][i] = data[i];
	}
	increment_buffer_ptr(rb);
}

void increment_buffer_ptr(struct RingBuffer* rb){
	if(rb->buffer_ptr < BUFFERSIZE - 1){
		(rb->buffer_ptr)++;
	} else if(rb->buffer_ptr == BUFFERSIZE - 1){
		rb->buffer_ptr = 0;
	} else {
		while(1){
			// buffer ptr out of range
		}
	}
}

void copy_inference_batch(struct RingBuffer* rb, int8_t* batch){

	for(int i = 0; i < BUFFERSIZE; i++){
		for(int j = 0; j < N_MFCC; j++){
			batch[i * N_MFCC + j] = rb->data[(i + rb->buffer_ptr) % BUFFERSIZE][j];
		}
	}

	update_last_inference_head(rb);
}

void update_last_inference_head(struct RingBuffer* rb){
	if (rb->buffer_ptr > 0 && rb->buffer_ptr < BUFFERSIZE){
		rb->last_inference_head = rb->buffer_ptr - 1;
	} else if (rb->buffer_ptr == 0){
		rb->last_inference_head = BUFFERSIZE - 1;
	} else {
		while(1){
			// while loop of shame
		}
	}
}
