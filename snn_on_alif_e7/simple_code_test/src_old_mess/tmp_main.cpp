/*
 * SPDX-FileCopyrightText: Copyright 2021-2023 Arm Limited and/or its
 * affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * This object detection example is intended to work with the
 * CMSIS pack produced by ml-embedded-eval-kit. The pack consists
 * of platform agnostic end-to-end ML use case API's that can be
 * used to construct ML examples for any target that can support
 * the memory requirements for TensorFlow-Lite-Micro framework and
 * some heap for the API runtime.
 */
 #include "BufAttributes.hpp" /* Buffer attributes to be applied */
 #include "Classifier.hpp"    /* Classifier for the result */
 #include "DetectorPostProcessing.hpp" /* Post Process */
 #include "DetectorPreProcessing.hpp"  /* Pre Process */      
 #include "DetectionResult.hpp"
 #include "YoloFastestModel.hpp"       /* Model API */
 #include "CameraCapture.hpp"          /* Live camera capture API */
 #include "LcdDisplay.hpp"             /* LCD display */
 #include "GpioSignal.hpp"             /* GPIO signals to drive LEDs */
 #include "InputFiles.hpp"
 
 /* Platform dependent files */
 #include "RTE_Components.h"  /* Provides definition for CMSIS_device_header */
 #include CMSIS_device_header /* Gives us IRQ num, base addresses. */
 #include "BoardInit.hpp"      /* Board initialisation */
 #include "log_macros.h"      /* Logging macros (optional) */
 #include "gpio_wrapper.h"    /* GPIO wrapper for LED control */
 #include "Driver_GPIO.h"     /* GPIO driver for LED control */
 
 #define CROPPED_IMAGE_WIDTH     192
 #define CROPPED_IMAGE_HEIGHT    192
 #define CROPPED_IMAGE_SIZE      (CROPPED_IMAGE_WIDTH * CROPPED_IMAGE_HEIGHT * 3)
 
 namespace arm {
 namespace app {
     /* Tensor arena buffer */
     static uint8_t tensorArena[ACTIVATION_BUF_SZ] ACTIVATION_BUF_ATTRIBUTE;
 
     /* RGB image buffer - cropped/scaled version of the original + debayered. */
     static uint8_t rgbImage[CROPPED_IMAGE_SIZE] __attribute__((section("rgb_buf"), aligned(16)));
 
     /* RAW image buffer. */
     static uint8_t rawImage[CAMERA_IMAGE_RAW_SIZE] __attribute__((section("raw_buf"), aligned(16)));
 
     /* LCD image buffer */
     static uint8_t lcdImage[DIMAGE_Y][DIMAGE_X][RGB_BYTES] __attribute__((section("lcd_buf"), aligned(16)));
 
     /* Optional getter function for the model pointer and its size. */
     namespace object_detection {
         extern uint8_t* GetModelPointer();
         extern size_t GetModelLen();
     } /* namespace object_detection */
 } /* namespace app */
 } /* namespace arm */
 
 typedef arm::app::object_detection::DetectionResult OdResults;
 
 /**
  * @brief Draws a boxes in the image using the object detection results vector.
  *
  * @param[out] rgbImage     Pointer to the start of the image.
  * @param[in]  width        Image width.
  * @param[in]  height       Image height.
  * @param[in]  results      Vector of object detection results.
  */
 static void DrawDetectionBoxes(uint8_t* rgbImage,
                                const uint32_t imageWidth,
                                const uint32_t imageHeight,
                                const std::vector<OdResults>& results);
 
 #if defined(__ARMCC_VERSION) && (__ARMCC_VERSION >= 6010050)
 __asm("  .global __ARM_use_no_argv\n");
 #endif
 
 #define _GET_DRIVER_REF(ref, peri, chan) \
     extern ARM_DRIVER_##peri Driver_##peri##chan; \
     static ARM_DRIVER_##peri * ref = &Driver_##peri##chan;
 #define GET_DRIVER_REF(ref, peri, chan) _GET_DRIVER_REF(ref, peri, chan)
 
 // Update to use the GPIO associated with P15_1 (W2)
 GET_DRIVER_REF(gpio_led, GPIO, 15);
 GET_DRIVER_REF(gpio_b, GPIO, BOARD_LEDRGB0_B_GPIO_PORT);
 GET_DRIVER_REF(gpio_r, GPIO, BOARD_LEDRGB0_R_GPIO_PORT);
 
 #define LED_PIN_NO  1
 
 int main()
 {
    
     printf("Starting object detection example...\n");
 
     gpio_b->Initialize(BOARD_LEDRGB0_B_PIN_NO, NULL);
     gpio_b->PowerControl(BOARD_LEDRGB0_B_PIN_NO, ARM_POWER_FULL);
     gpio_b->SetDirection(BOARD_LEDRGB0_B_PIN_NO, GPIO_PIN_DIRECTION_OUTPUT);
     gpio_b->SetValue(BOARD_LEDRGB0_B_PIN_NO, GPIO_PIN_OUTPUT_STATE_LOW);
 
     gpio_r->Initialize(BOARD_LEDRGB0_R_PIN_NO, NULL);
     gpio_r->PowerControl(BOARD_LEDRGB0_R_PIN_NO, ARM_POWER_FULL);
     gpio_r->SetDirection(BOARD_LEDRGB0_R_PIN_NO, GPIO_PIN_DIRECTION_OUTPUT);
     gpio_r->SetValue(BOARD_LEDRGB0_R_PIN_NO, GPIO_PIN_OUTPUT_STATE_LOW);
 
     gpio_led->Initialize(LED_PIN_NO, NULL);
     gpio_led->PowerControl(LED_PIN_NO, ARM_POWER_FULL);
     gpio_led->SetDirection(LED_PIN_NO, GPIO_PIN_DIRECTION_OUTPUT);
     gpio_led->SetValue(LED_PIN_NO, GPIO_PIN_OUTPUT_STATE_LOW); // Start with LED OFF
 
     /* Initialise the UART module to allow printf related functions (if using retarget) */
     BoardInit();
     
     /* Model object creation and initialisation. */
     arm::app::YoloFastestModel model;
     if (!model.Init(arm::app::tensorArena,
                     sizeof(arm::app::tensorArena),
                     arm::app::object_detection::GetModelPointer(),
                     arm::app::object_detection::GetModelLen())) {
         printf_err("Failed to initialise model\n");
         return 1;
     }
 
     printf("model.init() successful\n");
 
     auto initialImgIdx = 0;
 
     TfLiteTensor* inputTensor   = model.GetInputTensor(0);
     TfLiteTensor* outputTensor0 = model.GetOutputTensor(0);
     TfLiteTensor* outputTensor1 = model.GetOutputTensor(1);
 
     if (!inputTensor->dims) {
         printf_err("Invalid input tensor dims\n");
         return 1;
     } else if (inputTensor->dims->size < 3) {
         printf_err("Input tensor dimension should be >= 3\n");
         return 1;
     }
 
     printf("get IO tensors successful\n");
 
     TfLiteIntArray* inputShape = model.GetInputShape(0);
 
     const int inputImgCols = inputShape->data[arm::app::YoloFastestModel::ms_inputColsIdx];
     const int inputImgRows = inputShape->data[arm::app::YoloFastestModel::ms_inputRowsIdx];
 
     /* Set up pre and post-processing. */
     arm::app::DetectorPreProcess preProcess =
         arm::app::DetectorPreProcess(inputTensor, true, model.IsDataSigned());
 
 
         printf("prePrcess definition successful\n");
 
     std::vector<OdResults> results;
 
     const arm::app::object_detection::PostProcessParams postProcessParams{
         inputImgRows,
         inputImgCols,
         arm::app::object_detection::originalImageSize,
         arm::app::object_detection::anchor1,
         arm::app::object_detection::anchor2};
 
 
     printf("postProcessParams definition successful\n");
 
     
     arm::app::DetectorPostProcess postProcess =
         arm::app::DetectorPostProcess(outputTensor0, outputTensor1, results, postProcessParams);
 
     printf("postPrcess definition successful\n");
 
     /* Strings for presentation/logging. */
     std::string str_inf{"Running inference... "};
 
     const uint8_t* currImage = get_img_array(0);
 
      //while (1) {
         /* Infinite loop */
 
     auto dstPtr = static_cast<uint8_t*>(inputTensor->data.uint8);
     const size_t copySz =
         inputTensor->bytes < IMAGE_DATA_SIZE ? inputTensor->bytes : IMAGE_DATA_SIZE;
 
     /* Run the pre-processing, inference and post-processing. */
     if (!preProcess.DoPreProcess(currImage, copySz)) {
         printf_err("Pre-processing failed.");
         return 1;
     }
 
     /* Run inference over this image. */
     info("Running inference on image %" PRIu32 " => %s\n", 0, get_filename(0));
 
     //mydebug
     printf("pointer to instruction array: %p\n", arm::app::object_detection::GetModelPointer());
     printf("pointer to instruction array: %x\n", *arm::app::object_detection::GetModelPointer());
 
     if (!model.RunInference()) {
         printf_err("Inference failed.");
         return 2;
     }
 
     if (!postProcess.DoPostProcess()) {
         printf_err("Post-processing failed.");
         return 3;
     }
 
 
 
     //check tensors:
     
 
 
     /* Log the results. */
     for (uint32_t i = 0; i < results.size(); ++i) {
         printf("Detection at index %" PRIu32 ", at x-coordinate %" PRIu32 ", y-coordinate %" PRIu32
              ", width %" PRIu32 ", height %" PRIu32 "\n",
              i,
              results[i].m_x0,
              results[i].m_y0,
              results[i].m_w,
              results[i].m_h);
     }
 
     // DrawDetectionBoxes(arm::app::rgbImage, inputImgCols, inputImgRows, results);
 
     results.clear();
 
     gpio_led->SetValue(LED_PIN_NO, GPIO_PIN_OUTPUT_STATE_TOGGLE);
     gpio_b->SetValue(BOARD_LEDRGB0_B_PIN_NO, GPIO_PIN_OUTPUT_STATE_TOGGLE);
     
     //}
 
     return 0;
 
 }
 
 /**
  * @brief Draws a box in the image using the object detection result object.
  *
  * @param[out] imageData    Pointer to the start of the image.
  * @param[in]  width        Image width.
  * @param[in]  height       Image height.
  * @param[in]  result       Object detection result.
  */
 static void DrawBox(uint8_t* imageData,
                     const uint32_t width,
                     const uint32_t height,
                     const OdResults& result)
 {
     const auto x = result.m_x0;
     const auto y = result.m_y0;
     const auto w = result.m_w;
     const auto h = result.m_h;
 
     const uint32_t step = width * 3;
     uint8_t* const imStart = imageData + (y * step) + (x * 3);
 
     uint8_t* dst_0 = imStart;
     uint8_t* dst_1 = imStart + (h * step);
 
     for (uint32_t i = 0; i < w; ++i) {
         *dst_0 = 255;
         *dst_1 = 255;
 
         dst_0 += 3;
         dst_1 += 3;
     }
 
     dst_0 = imStart;
     dst_1 = imStart + (w * 3);
 
     for (uint32_t j = 0; j < h; ++j) {
         *dst_0 = 255;
         *dst_1 = 255;
 
         dst_0 += step;
         dst_1 += step;
     }
 }
 
 static void DrawDetectionBoxes(uint8_t* rgbImage,
                                const uint32_t imageWidth,
                                const uint32_t imageHeight,
                                const std::vector<OdResults>& results)
 {
     for (const auto& result : results) {
         DrawBox(rgbImage, imageWidth, imageHeight, result);
         printf("Detection :: [%" PRIu32 ", %" PRIu32
                          ", %" PRIu32 ", %" PRIu32 "]\n",
                 result.m_x0,
                 result.m_y0,
                 result.m_w,
                 result.m_h);
     }
 }
 