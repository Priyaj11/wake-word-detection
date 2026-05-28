// On-device MFCC feature extractor.
//
// Mirrors the Python implementation in python/features.py.  The
// algorithm:
//   1. Pre-emphasis (alpha = 0.97)
//   2. Hamming window over kFeatureWindowSize samples
//   3. Real FFT (length kFftSize) via CMSIS-DSP
//   4. Power spectrum
//   5. Mel filterbank (kMelBinCount triangular filters between
//      kFreqMin and kFreqMax)
//   6. log() on the filterbank energies
//   7. DCT-II to obtain kFeatureCount cepstral coefficients
//   8. Per-tensor int8 quantisation using the TFLite input parameters
//
// The mel filterbank and DCT matrix are computed once in ``Begin()`` and
// reused for every frame.
#include "feature_provider.h"

#include <Arduino.h>
#include <math.h>
#include <string.h>

#include "arm_math.h"  // CMSIS-DSP
#include "config.h"

namespace hey_jarvis {
namespace {

// Reusable scratch buffers — declared at namespace scope so they go in
// BSS rather than on the stack (the Cortex-M4 stack is only 16 KB).
float g_frame_buffer[kFeatureWindowSize];
float g_pre_emphasis_prev = 0.0f;

float g_fft_input[kFftSize];
float g_fft_output[kFftSize];
float g_power_spectrum[kFftSize / 2 + 1];

float g_mel_filterbank[kMelBinCount][kFftSize / 2 + 1];
float g_dct_matrix[kFeatureCount][kMelBinCount];

float g_hamming[kFeatureWindowSize];
arm_rfft_fast_instance_f32 g_fft_instance;

inline float HzToMel(float hz)   { return 2595.0f * log10f(1.0f + hz / 700.0f); }
inline float MelToHz(float mel)  { return 700.0f * (powf(10.0f, mel / 2595.0f) - 1.0f); }

void BuildHammingWindow() {
  for (int n = 0; n < kFeatureWindowSize; ++n) {
    g_hamming[n] = 0.54f - 0.46f * cosf(2.0f * PI * n / (kFeatureWindowSize - 1));
  }
}

void BuildMelFilterbank() {
  const float mel_min = HzToMel(kFreqMin);
  const float mel_max = HzToMel(kFreqMax);
  float points[kMelBinCount + 2];
  for (int i = 0; i < kMelBinCount + 2; ++i) {
    const float mel = mel_min + (mel_max - mel_min) * i / (kMelBinCount + 1);
    points[i] = MelToHz(mel);
  }
  // Convert Hz → FFT bin index
  int bins[kMelBinCount + 2];
  for (int i = 0; i < kMelBinCount + 2; ++i) {
    bins[i] = static_cast<int>(floorf((kFftSize + 1) * points[i] / kAudioSampleFrequency));
    if (bins[i] < 0) bins[i] = 0;
    if (bins[i] > kFftSize / 2) bins[i] = kFftSize / 2;
  }
  for (int m = 0; m < kMelBinCount; ++m) {
    for (int k = 0; k < kFftSize / 2 + 1; ++k) g_mel_filterbank[m][k] = 0.0f;
    const int left = bins[m];
    const int center = bins[m + 1];
    const int right = bins[m + 2];
    for (int k = left; k < center; ++k) {
      const float denom = static_cast<float>(center - left);
      if (denom > 0) g_mel_filterbank[m][k] = (k - left) / denom;
    }
    for (int k = center; k < right; ++k) {
      const float denom = static_cast<float>(right - center);
      if (denom > 0) g_mel_filterbank[m][k] = (right - k) / denom;
    }
  }
}

void BuildDctMatrix() {
  const float scale = sqrtf(2.0f / kMelBinCount);
  for (int n = 0; n < kFeatureCount; ++n) {
    for (int m = 0; m < kMelBinCount; ++m) {
      g_dct_matrix[n][m] = scale * cosf(PI * n * (2.0f * m + 1.0f) / (2.0f * kMelBinCount));
    }
  }
  // n == 0 → multiply by 1/sqrt(2) (standard DCT-II "ortho" norm)
  const float norm_zero = 1.0f / sqrtf(2.0f);
  for (int m = 0; m < kMelBinCount; ++m) {
    g_dct_matrix[0][m] *= norm_zero;
  }
}

}  // namespace

FeatureProvider::FeatureProvider() : initialised_(false) {}

bool FeatureProvider::Begin() {
  if (initialised_) return true;
  BuildHammingWindow();
  BuildMelFilterbank();
  BuildDctMatrix();
  if (arm_rfft_fast_init_f32(&g_fft_instance, kFftSize) != ARM_MATH_SUCCESS) {
    return false;
  }
  initialised_ = true;
  return true;
}

bool FeatureProvider::ExtractFeatures(const int16_t* audio, int audio_samples,
                                      float input_scale, int input_zero_point,
                                      int8_t* out_features) {
  if (!initialised_ || audio_samples < kClipSamples) return false;

  // Pre-emphasis applied once across the whole clip for consistency.
  static float pre_emphasised[kClipSamples];
  pre_emphasised[0] = static_cast<float>(audio[0]) / 32768.0f;
  for (int i = 1; i < kClipSamples; ++i) {
    const float cur = static_cast<float>(audio[i]) / 32768.0f;
    const float prev = static_cast<float>(audio[i - 1]) / 32768.0f;
    pre_emphasised[i] = cur - 0.97f * prev;
  }

  for (int frame = 0; frame < kFeatureSliceCount; ++frame) {
    const int start = frame * kFeatureWindowStride;
    // Window
    for (int n = 0; n < kFeatureWindowSize; ++n) {
      g_frame_buffer[n] = pre_emphasised[start + n] * g_hamming[n];
    }
    // Zero-pad to kFftSize
    memcpy(g_fft_input, g_frame_buffer, sizeof(float) * kFeatureWindowSize);
    memset(g_fft_input + kFeatureWindowSize, 0,
           sizeof(float) * (kFftSize - kFeatureWindowSize));

    // Real FFT
    arm_rfft_fast_f32(&g_fft_instance, g_fft_input, g_fft_output, 0);

    // Power spectrum: g_fft_output is packed (re0, re_N/2, re1, im1, ...).
    g_power_spectrum[0] = g_fft_output[0] * g_fft_output[0];
    g_power_spectrum[kFftSize / 2] = g_fft_output[1] * g_fft_output[1];
    for (int k = 1; k < kFftSize / 2; ++k) {
      const float re = g_fft_output[2 * k];
      const float im = g_fft_output[2 * k + 1];
      g_power_spectrum[k] = re * re + im * im;
    }

    // Mel filterbank + log
    float mel_energies[kMelBinCount];
    for (int m = 0; m < kMelBinCount; ++m) {
      float e = 0.0f;
      for (int k = 0; k < kFftSize / 2 + 1; ++k) {
        e += g_mel_filterbank[m][k] * g_power_spectrum[k];
      }
      mel_energies[m] = logf(e + 1e-6f);
    }

    // DCT-II → MFCC
    float mfcc[kFeatureCount];
    for (int n = 0; n < kFeatureCount; ++n) {
      float acc = 0.0f;
      for (int m = 0; m < kMelBinCount; ++m) {
        acc += g_dct_matrix[n][m] * mel_energies[m];
      }
      mfcc[n] = acc;
    }

    // Quantise to int8 using the TFLite input scale/zero-point.
    const int row = frame * kFeatureCount;
    for (int n = 0; n < kFeatureCount; ++n) {
      int32_t q = static_cast<int32_t>(roundf(mfcc[n] / input_scale)) + input_zero_point;
      if (q < -128) q = -128;
      if (q > 127) q = 127;
      out_features[row + n] = static_cast<int8_t>(q);
    }
  }
  return true;
}

}  // namespace hey_jarvis
