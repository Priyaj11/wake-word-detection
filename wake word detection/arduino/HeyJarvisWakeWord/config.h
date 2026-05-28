// Hey-Jarvis wake-word detector — shared on-device configuration.
//
// IMPORTANT: every value here must match the corresponding constant in
// python/config.py.  If you retrain with a different MFCC layout, edit
// both files together.
#pragma once

namespace hey_jarvis {

// ------------------- Audio -------------------
constexpr int kAudioSampleFrequency = 16000;      // Hz
constexpr int kClipDurationMs        = 1000;
constexpr int kClipSamples           = kAudioSampleFrequency * kClipDurationMs / 1000;

// ------------------- Feature extraction -------------------
constexpr int kFeatureWindowSizeMs   = 30;
constexpr int kFeatureWindowStrideMs = 20;
constexpr int kFeatureWindowSize     = kAudioSampleFrequency * kFeatureWindowSizeMs   / 1000;  // 480
constexpr int kFeatureWindowStride   = kAudioSampleFrequency * kFeatureWindowStrideMs / 1000;  // 320
constexpr int kFeatureCount          = 40;   // n_mfcc
constexpr int kFeatureSliceSize      = kFeatureCount;
constexpr int kFeatureSliceCount     = 49;   // (kClipSamples - kFeatureWindowSize) / kFeatureWindowStride + 1
constexpr int kFeatureElementCount   = kFeatureCount * kFeatureSliceCount;
constexpr int kFftSize               = 512;
constexpr int kMelBinCount           = 40;
constexpr float kFreqMin             = 20.0f;
constexpr float kFreqMax             = 4000.0f;

// ------------------- Model / inference -------------------
constexpr int kCategoryCount         = 2;
constexpr int kSilenceIndex          = 0;
constexpr int kHeyJarvisIndex        = 1;
constexpr const char* kCategoryLabels[kCategoryCount] = {"background", "hey_jarvis"};

// Tensor arena size — sized for the DS-CNN model.  Bump if the model grows.
constexpr int kTensorArenaSize       = 60 * 1024;

// ------------------- Sliding window / responder -------------------
constexpr int kAveragingWindowMs     = 1000;
constexpr int kSuppressionMs         = 1500;
constexpr int kMinDetectionsInWindow = 2;
constexpr int kDetectionThresholdQ8  = 230;   // 0.90 * 255 (uint8 probability)

// LED config
constexpr int kStatusLedPin          = LED_BUILTIN;
constexpr int kDetectionLedMs        = 800;

}  // namespace hey_jarvis
