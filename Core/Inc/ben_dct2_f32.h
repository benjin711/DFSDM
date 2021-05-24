/*
 * ben_dct2_f32.h
 *
 *  Created on: May 23, 2021
 *      Author: benjin
 */

#ifndef INC_BEN_DCT2_F32_H_
#define INC_BEN_DCT2_F32_H_

#include <arm_math.h>

void ben_dct2_f32(float32_t* pInlineBuffer, float32_t* pState, float32_t* mfcc_out, arm_rfft_fast_instance_f32* pRfft);

#endif /* INC_BEN_DCT2_F32_H_ */
