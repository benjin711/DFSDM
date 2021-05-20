/*
 * MFCC.c
 *
 *  Created on: May 16, 2021
 *      Author: benjin
 */

#include <arm_math.h>
#include <stdlib.h>
#include "MFCC.h"

void compute_mfccs_init_objects(struct compute_mfccs_objects* objs, struct compute_mfccs_params* params) {
	arm_rfft_fast_init_f32(objs->rfft_struct, params->frame_length);
	arm_cfft_init_f32(objs->cfft_struct, params->frame_length);
}

void compute_mfccs(float* input_vector, float* mfccs, struct compute_mfccs_objects* objs, struct compute_mfccs_params* params, float* a, float* b) {
	// Prepare b as input vector to the cfft
	for(int i=0; i < params->frame_length; i++){
		b[i*2] = input_vector[i];
	}

	// RFFT
	//float* rfft = (float*) calloc(params->frame_length, sizeof(float));
	arm_rfft_fast_f32(objs->rfft_struct, input_vector, a, 0);

	// CFFT
	//float* cfft = (float*) calloc(2*params->frame_length, sizeof(float));
	arm_cfft_f32(objs->cfft_struct, b, 0, 0);
}
