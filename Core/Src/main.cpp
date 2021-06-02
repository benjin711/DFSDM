/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.c
  * @brief          : Main program body
  ******************************************************************************
  * @attention
  *
  * <h2><center>&copy; Copyright (c) 2019 STMicroelectronics.
  * All rights reserved.</center></h2>
  *
  * This software component is licensed by ST under BSD 3-Clause license,
  * the "License"; You may not use this file except in compliance with the
  * License. You may obtain a copy of the License at:
  *                        opensource.org/licenses/BSD-3-Clause
  *
  ******************************************************************************
  */
/* USER CODE END Header */
/* Includes ------------------------------------------------------------------*/
#include "main.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */
#include <string.h>

#include <CycleCounter.h>
#include <stdio.h>
#include <arm_math.h>
#include "MFCC09.h"
#include "linear_to_mel_weight_list.h"
#include "ben_dct2_f32.h"
#include "ring_buffer.h"

#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/version.h"

#define ARM_MATH_CM4
#define ARM_MATH_DSP
#define CMSIS_NN
#define TF_LITE_USE_GLOBAL_CMATH_FUNCTIONS
#define TF_LITE_USE_GLOBAL_MAX
#define TF_LITE_USE_GLOBAL_MIN

#define QUEUELENGTH 2048
#define SYSCLK 80000000
#define SAMPLINGRATE 9524
#define N_MFCCS 13
#define INPUT_SCALE 0.003135847859084606
#define INPUT_ZERO_POINT -128
/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */

/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */

/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */

/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/
DFSDM_Filter_HandleTypeDef hdfsdm1_filter0;
DFSDM_Channel_HandleTypeDef hdfsdm1_channel2;
DMA_HandleTypeDef hdma_dfsdm1_flt0;
USART_HandleTypeDef husart1;

/* USER CODE BEGIN PV */
namespace {
  tflite::ErrorReporter* error_reporter = nullptr;
  const tflite::Model* model = nullptr;
  tflite::MicroInterpreter* interpreter = nullptr;
  TfLiteTensor* model_input = nullptr;
  TfLiteTensor* model_output = nullptr;
} // namespace

/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
static void MX_GPIO_Init(void);
static void MX_DMA_Init(void);
static void MX_DFSDM1_Init(void);
static void MX_USART1_Init(void);
/* USER CODE BEGIN PFP */

/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN 0 */

volatile bool firstHalfFull = false;
volatile bool secondHalfFull = false;

int32_t RecBuff[QUEUELENGTH];
int16_t amplitude;

#ifdef __GNUC__
/* With GCC/RAISONANCE, small msg_info (option LD Linker->Libraries->Small msg_info
   set to 'Yes') calls __io_putchar() */
#define PUTCHAR_PROTOTYPE int __io_putchar(int ch)
#define GETCHAR_PROTOTYPE int __io_getchar(void)
#else
#define PUTCHAR_PROTOTYPE int fputc(int ch, FILE *f)
#define GETCHAR_PROTOTYPE int fgetc(FILE *f)
#endif /* __GNUC__ */

PUTCHAR_PROTOTYPE
{
  /* Place your implementation of fputc here */
  /* e.g. write a character to the serial port and Loop until the end of transmission */
  while (HAL_OK != HAL_USART_Transmit(&husart1, (uint8_t *) &ch, 1, 30000))
  {
    ;
  }
  return ch;
}

/**
  * @brief Retargets the C library scanf function to the USART.
  * @param None
  * @retval None
  */
GETCHAR_PROTOTYPE
{
  /* Place your implementation of fgetc here */
  /* e.g. readwrite a character to the USART2 and Loop until the end of transmission */
  uint8_t ch = 0;
  while (HAL_OK != HAL_USART_Receive(&husart1, (uint8_t *)&ch, 1, 30000))
  {
    ;
  }
  return ch;
}


void HAL_DFSDM_FilterRegConvCpltCallback(DFSDM_Filter_HandleTypeDef *hdfsdm_filter)
{
	firstHalfFull = true;
}

void HAL_DFSDM_FilterRegConvHalfCpltCallback(DFSDM_Filter_HandleTypeDef *hdfsdm_filter)
{
	secondHalfFull = true;
}

/**
  * @brief Convert the float mfccs into int8 after normalization
  * @param mfccs_float, mfccs_int8
  * @retval None
  */
void normalize_mfccs(float32_t* mfccs_float, int8_t* mfccs_int8){
	for(int i = 0; i<N_MFCCS; i++){
		mfccs_int8[i] = (int8_t)((mfccs_float[i] / 512 + 0.5) / INPUT_SCALE + INPUT_ZERO_POINT);
	}
}



/* USER CODE END 0 */

/**
  * @brief  The application entry point.
  * @retval int
  */
int main(void)
{
  /* USER CODE BEGIN 1 */
	char buf[50];
	int buf_len = 0;
	TfLiteStatus tflite_status;
	uint32_t num_elements;
	uint32_t num_output_elements;
	int8_t output[2];
	const int kTensorArenaSize = 30 * 1024;
	static uint8_t tensor_arena[kTensorArenaSize];

  /* USER CODE END 1 */

  /* MCU Configuration--------------------------------------------------------*/

  /* Reset of all peripherals, Initializes the Flash interface and the Systick. */
  HAL_Init();

  /* USER CODE BEGIN Init */

  /* USER CODE END Init */

  /* Configure the system clock */
  SystemClock_Config();

  /* USER CODE BEGIN SysInit */

  /* USER CODE END SysInit */

  /* Initialize all configured peripherals */
  MX_GPIO_Init();
  MX_DMA_Init();
  MX_DFSDM1_Init();
  MX_USART1_Init();
  /* USER CODE BEGIN 2 */
	if(HAL_OK != HAL_DFSDM_FilterRegularStart_DMA(&hdfsdm1_filter0, RecBuff, QUEUELENGTH)){
    Error_Handler();
  }

	static tflite::MicroErrorReporter micro_error_reporter;
	error_reporter = &micro_error_reporter;
	// Say something to test error reporter
	buf_len = sprintf(buf, "START TEST\r\n");
	HAL_USART_Transmit(&husart1, (uint8_t *)buf, buf_len, 100);
	error_reporter->Report("STM32 TensorFlow Lite test");
	// Map the model into a usable data structure
	model = tflite::GetModel(MFCC);
	if (model->version() != TFLITE_SCHEMA_VERSION)
	{
		error_reporter->Report("Model version does not match Schema");
	while(1);
	}

	tflite::AllOpsResolver micro_op_resolver;
	//tflite::MicroMutableOpResolver<16> micro_op_resolver;
//	tflite_status = micro_op_resolver.AddConv2D();
//	if (tflite_status != kTfLiteOk)
//	{
//		error_reporter->Report("Could not add CONV2D op");
//		while(1);
//	}
//	tflite_status = micro_op_resolver.AddRelu();
//	if (tflite_status != kTfLiteOk)
//	{
//		error_reporter->Report("Could not add RELU op");
//		while(1);
//	}
//	tflite_status = micro_op_resolver.AddConv2D();
//	if (tflite_status != kTfLiteOk)
//	{
//		error_reporter->Report("Could not add CONV2D op");
//		while(1);
//	}
//	tflite_status = micro_op_resolver.AddRelu();
//	if (tflite_status != kTfLiteOk)
//	{
//		error_reporter->Report("Could not add RELU op");
//		while(1);
//	}
//	tflite_status = micro_op_resolver.AddMaxPool2D();
//	if (tflite_status != kTfLiteOk)
//	{
//		error_reporter->Report("Could not add RELU op");
//		while(1);
//	}
//	tflite_status = micro_op_resolver.AddConv2D();
//	if (tflite_status != kTfLiteOk)
//	{
//		error_reporter->Report("Could not add CONV2D op");
//		while(1);
//	}
//	tflite_status = micro_op_resolver.AddRelu();
//	if (tflite_status != kTfLiteOk)
//	{
//		error_reporter->Report("Could not add RELU op");
//		while(1);
//	}
//	tflite_status = micro_op_resolver.AddMaxPool2D();
//	if (tflite_status != kTfLiteOk)
//	{
//		error_reporter->Report("Could not add RELU op");
//		while(1);
//	}
//	tflite_status = micro_op_resolver.AddConv2D();
//	if (tflite_status != kTfLiteOk)
//	{
//		error_reporter->Report("Could not add CONV2D op");
//		while(1);
//	}
//	tflite_status = micro_op_resolver.AddRelu();
//	if (tflite_status != kTfLiteOk)
//	{
//		error_reporter->Report("Could not add RELU op");
//		while(1);
//	}
//	tflite_status = micro_op_resolver.AddMean();
//	if (tflite_status != kTfLiteOk)
//	{
//		error_reporter->Report("Could not add MEAN op");
//		while(1);
//	}
//	tflite_status =micro_op_resolver.AddReshape();
//	if (tflite_status != kTfLiteOk)
//	{
//		error_reporter->Report("Could not add RESHAPE op");
//		while(1);
//	}
//	tflite_status = micro_op_resolver.AddFullyConnected();
//	if (tflite_status != kTfLiteOk)
//	{
//	error_reporter->Report("Could not add FULLY CONNECTED op");
//	while(1);
//	}
//	tflite_status = micro_op_resolver.AddRelu();
//	if (tflite_status != kTfLiteOk)
//	{
//		error_reporter->Report("Could not add RELU op");
//		while(1);
//	}
//	tflite_status = micro_op_resolver.AddFullyConnected();
//	if (tflite_status != kTfLiteOk)
//	{
//	error_reporter->Report("Could not add FULLY CONNECTED op");
//	while(1);
//	}
//	tflite_status = micro_op_resolver.AddSoftmax();
//	if (tflite_status != kTfLiteOk)
//	{
//		error_reporter->Report("Could not add Softmax op");
//		while(1);
//	}

	static tflite::MicroInterpreter static_interpreter(
		model, micro_op_resolver, tensor_arena, kTensorArenaSize, error_reporter);
	interpreter = &static_interpreter;

	tflite_status = interpreter->AllocateTensors();

	if (tflite_status != kTfLiteOk)
	{
		buf_len = sprintf(buf, "Failed tensors\r\n");
		HAL_USART_Transmit(&husart1, (uint8_t *)buf, buf_len, 100);
		error_reporter->Report("AllocateTensors() failed");
		while(1);
	}

	// Assign model input and output buffers (tensors) to pointers
	model_input = interpreter->input(0);
	model_output = interpreter->output(0);
	float input_size = model_input->dims->size;
	buf_len = sprintf(buf, "Model input size: %f\r\n", input_size);
	HAL_USART_Transmit(&husart1, (uint8_t *)buf, buf_len, 100);
	// Get number of elements in input tensor
	num_elements = model_input->bytes;
	buf_len = sprintf(buf, "Number of input elements: %lu\r\n", num_elements);
	HAL_USART_Transmit(&husart1, (uint8_t *)buf, buf_len, 100);



  /* USER CODE END 2 */

  /* Infinite loop */
  /* USER CODE BEGIN WHILE */


	// Parameters
	const int sr = SAMPLINGRATE;
	const int fl = QUEUELENGTH / 2; // frame length
	const int n_mb = 64; // mel bins
	const int n_mfccs = N_MFCCS;

	// Buffer
	float32_t buffer1[fl];
	float32_t buffer2[fl];
	struct RingBuffer rb;
	// Output
	float32_t mfccs_float[N_MFCCS];
	int8_t mfccs_int8[N_MFCCS];

	// Other Objects
	arm_rfft_fast_instance_f32 rfft_struct_v1;
	arm_rfft_fast_instance_f32 rfft_struct_v2;


	// Initialize Objects
	init_ring_buffer(&rb);
	arm_status status_v1 = arm_rfft_fast_init_f32(&rfft_struct_v1, fl);
	arm_status status_v2 = arm_rfft_fast_init_f32(&rfft_struct_v2, n_mb);


	// Debug
	bool flag = true;


  while (1)
  {
    /* USER CODE END WHILE */

    /* USER CODE BEGIN 3 */
		if(firstHalfFull && flag){
			for(int i=0;i<QUEUELENGTH/2;i++){
				buffer1[i] = (float32_t)(RecBuff[i]>>8);
//		  	buf_len = sprintf(buf, "%f\r\n", buffer1[i]);
//		  	HAL_USART_Transmit(&husart1, (uint8_t *)buf, buf_len, 100);
			}

			arm_rfft_fast_f32(&rfft_struct_v1, buffer1, buffer2, 0);
			arm_cmplx_mag_f32(buffer2, buffer1, fl / 2);
			calc_log_mel_spectrogram(buffer1, buffer2);
			ben_dct2_f32(buffer2, buffer1,  mfccs_float, &rfft_struct_v2);
			normalize_mfccs(mfccs_float, mfccs_int8);
			insert_data(&rb, mfccs_int8);

			firstHalfFull = false;
			//flag = false;
		}
		if(secondHalfFull){
			for(int i=QUEUELENGTH/2;i<QUEUELENGTH;i++){
				buffer1[i - QUEUELENGTH/2] = (float32_t)(RecBuff[i]>>8);
//				buf_len = sprintf(buf, "%f\r\n", buffer1[i - QUEUELENGTH/2]);
//				HAL_USART_Transmit(&husart1, (uint8_t *)buf, buf_len, 100);
			}

			arm_rfft_fast_f32(&rfft_struct_v1, buffer1, buffer2, 0);
			arm_cmplx_mag_f32(buffer2, buffer1, fl / 2);
			calc_log_mel_spectrogram(buffer1, buffer2);
			ben_dct2_f32(buffer2, buffer1,  mfccs_float, &rfft_struct_v2);
			normalize_mfccs(mfccs_float, mfccs_int8);
			insert_data(&rb, mfccs_int8);

			secondHalfFull = false;
		}

		if(rb.buffer_ptr >= rb.last_inference_head + 1){
			copy_inference_batch(&rb, model_input->data.int8);
			tflite_status = interpreter->Invoke();
			if(tflite_status != kTfLiteOk)
			{
				error_reporter->Report("Invoke failed");
			}
			output[0] = model_output->data.int8[0];
			output[1] = model_output->data.int8[1];
			num_output_elements = model_output->bytes;
			buf_len = sprintf(buf, "Number of output elements: %lu\r\n", num_output_elements);
			HAL_USART_Transmit(&husart1, (uint8_t *)buf, buf_len, 100);
			buf_len = sprintf(buf, "Output: [%d, %d]\r\n", output[0], output[1]);
			HAL_USART_Transmit(&husart1, (uint8_t *)buf, buf_len, 100);
		}

	}





  /* USER CODE END 3 */
}

/**
  * @brief System Clock Configuration
  * @retval None
  */
void SystemClock_Config(void)
{
  RCC_OscInitTypeDef RCC_OscInitStruct = {0};
  RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};
  RCC_PeriphCLKInitTypeDef PeriphClkInit = {0};

  /** Initializes the RCC Oscillators according to the specified parameters
  * in the RCC_OscInitTypeDef structure.
  */
  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_MSI;
  RCC_OscInitStruct.MSIState = RCC_MSI_ON;
  RCC_OscInitStruct.MSICalibrationValue = 0;
  RCC_OscInitStruct.MSIClockRange = RCC_MSIRANGE_6;
  RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
  RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_MSI;
  RCC_OscInitStruct.PLL.PLLM = 1;
  RCC_OscInitStruct.PLL.PLLN = 40;
  RCC_OscInitStruct.PLL.PLLP = RCC_PLLP_DIV7;
  RCC_OscInitStruct.PLL.PLLQ = RCC_PLLQ_DIV2;
  RCC_OscInitStruct.PLL.PLLR = RCC_PLLR_DIV2;
  if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
  {
    Error_Handler();
  }
  /** Initializes the CPU, AHB and APB buses clocks
  */
  RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK|RCC_CLOCKTYPE_SYSCLK
                              |RCC_CLOCKTYPE_PCLK1|RCC_CLOCKTYPE_PCLK2;
  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
  RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV1;
  RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV2;

  if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_4) != HAL_OK)
  {
    Error_Handler();
  }
  PeriphClkInit.PeriphClockSelection = RCC_PERIPHCLK_USART1|RCC_PERIPHCLK_SAI1
                              |RCC_PERIPHCLK_DFSDM1;
  PeriphClkInit.Usart1ClockSelection = RCC_USART1CLKSOURCE_SYSCLK;
  PeriphClkInit.Sai1ClockSelection = RCC_SAI1CLKSOURCE_PLLSAI1;
  PeriphClkInit.Dfsdm1ClockSelection = RCC_DFSDM1CLKSOURCE_PCLK;
  PeriphClkInit.PLLSAI1.PLLSAI1Source = RCC_PLLSOURCE_MSI;
  PeriphClkInit.PLLSAI1.PLLSAI1M = 1;
  PeriphClkInit.PLLSAI1.PLLSAI1N = 16;
  PeriphClkInit.PLLSAI1.PLLSAI1P = RCC_PLLP_DIV7;
  PeriphClkInit.PLLSAI1.PLLSAI1Q = RCC_PLLQ_DIV2;
  PeriphClkInit.PLLSAI1.PLLSAI1R = RCC_PLLR_DIV2;
  PeriphClkInit.PLLSAI1.PLLSAI1ClockOut = RCC_PLLSAI1_SAI1CLK;
  if (HAL_RCCEx_PeriphCLKConfig(&PeriphClkInit) != HAL_OK)
  {
    Error_Handler();
  }
  /** Configure the main internal regulator output voltage
  */
  if (HAL_PWREx_ControlVoltageScaling(PWR_REGULATOR_VOLTAGE_SCALE1) != HAL_OK)
  {
    Error_Handler();
  }
}

/**
  * @brief DFSDM1 Initialization Function
  * @param None
  * @retval None
  */
static void MX_DFSDM1_Init(void)
{

  /* USER CODE BEGIN DFSDM1_Init 0 */

  /* USER CODE END DFSDM1_Init 0 */

  /* USER CODE BEGIN DFSDM1_Init 1 */

  /* USER CODE END DFSDM1_Init 1 */
  hdfsdm1_filter0.Instance = DFSDM1_Filter0;
  hdfsdm1_filter0.Init.RegularParam.Trigger = DFSDM_FILTER_SW_TRIGGER;
  hdfsdm1_filter0.Init.RegularParam.FastMode = ENABLE;
  hdfsdm1_filter0.Init.RegularParam.DmaMode = ENABLE;
  hdfsdm1_filter0.Init.FilterParam.SincOrder = DFSDM_FILTER_SINC3_ORDER;
  hdfsdm1_filter0.Init.FilterParam.Oversampling = 64;
  hdfsdm1_filter0.Init.FilterParam.IntOversampling = 1;
  if (HAL_DFSDM_FilterInit(&hdfsdm1_filter0) != HAL_OK)
  {
    Error_Handler();
  }
  hdfsdm1_channel2.Instance = DFSDM1_Channel2;
  hdfsdm1_channel2.Init.OutputClock.Activation = ENABLE;
  hdfsdm1_channel2.Init.OutputClock.Selection = DFSDM_CHANNEL_OUTPUT_CLOCK_AUDIO;
  hdfsdm1_channel2.Init.OutputClock.Divider = 15;
  hdfsdm1_channel2.Init.Input.Multiplexer = DFSDM_CHANNEL_EXTERNAL_INPUTS;
  hdfsdm1_channel2.Init.Input.DataPacking = DFSDM_CHANNEL_STANDARD_MODE;
  hdfsdm1_channel2.Init.Input.Pins = DFSDM_CHANNEL_SAME_CHANNEL_PINS;
  hdfsdm1_channel2.Init.SerialInterface.Type = DFSDM_CHANNEL_SPI_RISING;
  hdfsdm1_channel2.Init.SerialInterface.SpiClock = DFSDM_CHANNEL_SPI_CLOCK_INTERNAL;
  hdfsdm1_channel2.Init.Awd.FilterOrder = DFSDM_CHANNEL_FASTSINC_ORDER;
  hdfsdm1_channel2.Init.Awd.Oversampling = 1;
  hdfsdm1_channel2.Init.Offset = 0;
  hdfsdm1_channel2.Init.RightBitShift = 0x03;
  if (HAL_DFSDM_ChannelInit(&hdfsdm1_channel2) != HAL_OK)
  {
    Error_Handler();
  }
  if (HAL_DFSDM_FilterConfigRegChannel(&hdfsdm1_filter0, DFSDM_CHANNEL_2, DFSDM_CONTINUOUS_CONV_ON) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN DFSDM1_Init 2 */

  /* USER CODE END DFSDM1_Init 2 */

}

/**
  * @brief USART1 Initialization Function
  * @param None
  * @retval None
  */
static void MX_USART1_Init(void)
{

  /* USER CODE BEGIN USART1_Init 0 */

  /* USER CODE END USART1_Init 0 */

  /* USER CODE BEGIN USART1_Init 1 */

  /* USER CODE END USART1_Init 1 */
  husart1.Instance = USART1;
  husart1.Init.BaudRate = 115200;
  husart1.Init.WordLength = USART_WORDLENGTH_8B;
  husart1.Init.StopBits = USART_STOPBITS_1;
  husart1.Init.Parity = USART_PARITY_NONE;
  husart1.Init.Mode = USART_MODE_TX_RX;
  husart1.Init.CLKPolarity = USART_POLARITY_LOW;
  husart1.Init.CLKPhase = USART_PHASE_1EDGE;
  husart1.Init.CLKLastBit = USART_LASTBIT_DISABLE;
  if (HAL_USART_Init(&husart1) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN USART1_Init 2 */

  /* USER CODE END USART1_Init 2 */

}

/**
  * Enable DMA controller clock
  */
static void MX_DMA_Init(void)
{

  /* DMA controller clock enable */
  __HAL_RCC_DMA1_CLK_ENABLE();

  /* DMA interrupt init */
  /* DMA1_Channel4_IRQn interrupt configuration */
  HAL_NVIC_SetPriority(DMA1_Channel4_IRQn, 0, 0);
  HAL_NVIC_EnableIRQ(DMA1_Channel4_IRQn);

}

/**
  * @brief GPIO Initialization Function
  * @param None
  * @retval None
  */
static void MX_GPIO_Init(void)
{

  /* GPIO Ports Clock Enable */
  __HAL_RCC_GPIOE_CLK_ENABLE();
  __HAL_RCC_GPIOA_CLK_ENABLE();
  __HAL_RCC_GPIOB_CLK_ENABLE();

}

/* USER CODE BEGIN 4 */

/* USER CODE END 4 */

/**
  * @brief  This function is executed in case of error occurrence.
  * @retval None
  */
void Error_Handler(void)
{
  /* USER CODE BEGIN Error_Handler_Debug */
  /* User can add his own implementation to report the HAL error return state */

  /* USER CODE END Error_Handler_Debug */
}

#ifdef  USE_FULL_ASSERT
/**
  * @brief  Reports the name of the source file and the source line number
  *         where the assert_param error has occurred.
  * @param  file: pointer to the source file name
  * @param  line: assert_param error line source number
  * @retval None
  */
void assert_failed(uint8_t *file, uint32_t line)
{
  /* USER CODE BEGIN 6 */
  /* User can add his own implementation to report the file name and line number,
     tex: printf("Wrong parameters value: file %s on line %d\r\n", file, line) */
  /* USER CODE END 6 */
}
#endif /* USE_FULL_ASSERT */

/************************ (C) COPYRIGHT STMicroelectronics *****END OF FILE****/
