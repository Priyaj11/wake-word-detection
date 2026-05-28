// Converts a kClipSamples-long PCM buffer into a flat array of int8
// MFCC features ready to be copied into the TFLite input tensor.
//
// The output layout matches python/features.py: row-major
// (kFeatureSliceCount, kFeatureSliceSize) → length
// kFeatureSliceCount * kFeatureSliceSize.
#pragma once

#include <stdint.h>

namespace hey_jarvis {

class FeatureProvider {
 public:
  FeatureProvider();
  bool Begin();
  // ``input_scale`` / ``input_zero_point`` come from the TFLite model's
  // quantization parameters and are applied as the final quantisation
  // step inside the provider.
  bool ExtractFeatures(const int16_t* audio, int audio_samples,
                       float input_scale, int input_zero_point,
                       int8_t* out_features);

 private:
  bool initialised_;
};

}  // namespace hey_jarvis
