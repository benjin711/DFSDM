#include <stdio.h>
#include "stm32l4xx_hal.h"

extern USART_HandleTypeDef husart1;

extern "C" void DebugLog(const char* s) {
	int buf_len = 0;
	char buf[50];
	buf_len = sprintf(buf, s);
	HAL_USART_Transmit(&husart1, (uint8_t *)buf, buf_len, 100);
}
