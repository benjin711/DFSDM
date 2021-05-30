/*
 * MFCC.h
 *
 *  Created on: May 16, 2021
 *      Author: benjin
 */

#ifndef INC_MFCC_H_
#define INC_MFCC_H_

#include <arm_math.h>

struct compute_mfccs_params {
	uint32_t sr;
	uint32_t frame_length;
	uint8_t n_mfccs;
};

struct compute_mfccs_objects {
	arm_rfft_fast_instance_f32* rfft_struct;
	arm_cfft_instance_f32* cfft_struct;
};

void compute_mfccs_init_objects(struct compute_mfccs_objects* objs, struct compute_mfccs_params* params);

void compute_mfccs(float* input_vector, float* mfccs, struct compute_mfccs_objects* objs, struct compute_mfccs_params* params, float* a, float* b);

#endif /* INC_MFCC_H_ */
