/*
 * Hey-Jarvis wake-word detector for the Arduino Nano 33 BLE Sense.
 *
 * Pipeline
 *  PDM mic → ring buffer → MFCC (CMSIS-DSP) → TFLite Micro inference →
 *  sliding-window decision → LED + Serial.
 *
 * Hardware
 *  - Arduino Nano 33 BLE Sense (nRF52840, Cortex-M4F)
 *  - Onboard MP34DT05 PDM microphone
 *  - Onboard LED (LED_BUILTIN)
 *
 * Libraries (install from the Arduino Library Manager)
 *  - Arduino_TensorFlowLite (Pete Warden / Arduino fork) — provides
 *    tensorflow/lite/micro/* and a CMSIS-DSP build for Cortex-M.
 *  - PDM (bundled with the mbed-based Nano 33 BLE Sense core).
 *  - Arduino_CMSIS-DSP (or equivalent providing arm_math.h).
 *
 * Build
 *  - Board: "Arduino Nano 33 BLE"
 *  - Tools → Optimize: "Smallest (-Os)"
 */

#include <Arduino.h>
#include <TensorFlowLite.h>

#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "audio_provider.h"
#include "command_responder.h"
#include "config.h"
#include "feature_provider.h"
#include "model.h"
#include "recognize_commands.h"

using hey_jarvis::AudioProviderBegin;
using hey_jarvis::AudioProviderGetSamples;
using hey_jarvis::AudioProviderOverruns;
using hey_jarvis::CommandResponderBegin;
using hey_jarvis::CommandResponderOnWakeWord;
using hey_jarvis::CommandResponderTick;
using hey_jarvis::FeatureProvider;
using hey_jarvis::RecognizeCommands;

namespace {

// TFLite globals.
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input_tensor = nullptr;
TfLiteTensor* output_tensor = nullptr;

// Tensor arena — keep it large enough for the DS-CNN model.
alignas(16) uint8_t tensor_arena[hey_jarvis::kTensorArenaSize];

// Buffers used inside the loop.
int16_t audio_buffer[hey_jarvis::kClipSamples];
FeatureProvider feature_provider;
RecognizeCommands recognizer;

// Stats.
uint32_t total_inferences = 0;
uint32_t last_status_print_ms = 0;

void FatalError(const __FlashStringHelper* msg) {
  while (true) {
    Serial.print(F("[FATAL] "));
    Serial.println(msg);
    digitalWrite(LED_BUILTIN, HIGH);
    delay(150);
    digitalWrite(LED_BUILTIN, LOW);
    delay(150);
  }
}

}  // namespace

void setup() {
  Serial.begin(115200);
  // Wait briefly for the host serial monitor — don't block forever on
  // battery-powered builds.
  const uint32_t serial_deadline = millis() + 3000;
  while (!Serial && millis() < serial_deadline) { delay(10); }

  Serial.println();
  Serial.println(F("================================================"));
  Serial.println(F(" Hey-Jarvis wake-word detector — Nano 33 BLE Sense"));
  Serial.println(F("================================================"));

  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  // -------- Load model --------
  model = tflite::GetModel(g_model);
  if (model == nullptr || g_model_len < 16) {
    FatalError(F("model not flashed — run python/convert.py and re-upload"));
  }
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.print(F("Schema mismatch — model "));
    Serial.print(model->version());
    Serial.print(F(" vs runtime "));
    Serial.println(TFLITE_SCHEMA_VERSION);
    FatalError(F("incompatible TFLite schema"));
  }

  // -------- Build interpreter --------
  static tflite::AllOpsResolver resolver;
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, hey_jarvis::kTensorArenaSize,
      error_reporter);
  interpreter = &static_interpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    FatalError(F("AllocateTensors() failed — increase kTensorArenaSize"));
  }
  input_tensor = interpreter->input(0);
  output_tensor = interpreter->output(0);

  // Sanity check tensor shape: expecting [1, kFeatureSliceCount, kFeatureSliceSize, 1]
  if (input_tensor->dims->size != 4 ||
      input_tensor->dims->data[1] != hey_jarvis::kFeatureSliceCount ||
      input_tensor->dims->data[2] != hey_jarvis::kFeatureSliceSize ||
      input_tensor->type != kTfLiteInt8) {
    FatalError(F("unexpected input tensor shape/type"));
  }

  // -------- Feature extractor --------
  if (!feature_provider.Begin()) {
    FatalError(F("FeatureProvider init failed"));
  }

  // -------- Audio capture --------
  if (!AudioProviderBegin()) {
    FatalError(F("PDM init failed — check Nano 33 BLE Sense board support"));
  }

  CommandResponderBegin();

  Serial.print(F("Model size: "));
  Serial.print(g_model_len);
  Serial.println(F(" bytes"));
  Serial.print(F("Arena   :  "));
  Serial.print(hey_jarvis::kTensorArenaSize);
  Serial.println(F(" bytes"));
  Serial.print(F("Threshold (uint8): "));
  Serial.println(hey_jarvis::kDetectionThresholdQ8);
  Serial.println(F("Listening for 'Hey Jarvis'..."));
}

void loop() {
  const uint32_t now = millis();
  CommandResponderTick(now);

  uint32_t timestamp = 0;
  if (!AudioProviderGetSamples(audio_buffer, &timestamp)) {
    // Not enough audio yet — keep the loop tight.
    return;
  }

  // -------- Feature extraction --------
  const float input_scale = input_tensor->params.scale;
  const int input_zero_point = input_tensor->params.zero_point;
  if (!feature_provider.ExtractFeatures(audio_buffer, hey_jarvis::kClipSamples,
                                        input_scale, input_zero_point,
                                        input_tensor->data.int8)) {
    Serial.println(F("[warn] feature extraction failed"));
    return;
  }

  // -------- Inference --------
  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.println(F("[warn] invoke failed"));
    return;
  }
  ++total_inferences;

  // -------- Read output --------
  // output_tensor is int8 with quantization (scale, zp).  Convert to a
  // 0–255 score for the positive class.
  const float out_scale = output_tensor->params.scale;
  const int out_zp = output_tensor->params.zero_point;
  const int8_t raw_pos = output_tensor->data.int8[hey_jarvis::kHeyJarvisIndex];
  const float prob = (raw_pos - out_zp) * out_scale;  // 0..1
  const int p = static_cast<int>(prob * 255.0f + 0.5f);
  const uint8_t pos_score = static_cast<uint8_t>(p < 0 ? 0 : (p > 255 ? 255 : p));

  bool found = false;
  uint8_t smoothed = 0;
  recognizer.ProcessLatestResult(pos_score, timestamp, &found, &smoothed);

  if (found) {
    CommandResponderOnWakeWord(smoothed, timestamp);
  }

  // -------- Periodic status (every 5 s) --------
  if (now - last_status_print_ms > 5000) {
    last_status_print_ms = now;
    Serial.print(F("[status] inferences="));
    Serial.print(total_inferences);
    Serial.print(F(" overruns="));
    Serial.print(AudioProviderOverruns());
    Serial.print(F(" last_score(q8)="));
    Serial.print(pos_score);
    Serial.print(F(" smoothed(q8)="));
    Serial.println(smoothed);
  }
}
