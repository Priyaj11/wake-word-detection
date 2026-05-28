// Sliding-window post-processor that smooths model outputs and decides
// when a wake word has been detected.
//
// Behaviour:
//   * Each inference's softmax probabilities (already converted to a
//     0–255 uint8 score) are pushed into a small ring of recent results.
//   * Scores older than ``kAveragingWindowMs`` are dropped.
//   * If the average positive-class score exceeds ``kDetectionThresholdQ8``
//     AND we have observed at least ``kMinDetectionsInWindow`` frames,
//     a detection is reported.
//   * After a detection, further reports are suppressed for
//     ``kSuppressionMs`` to avoid spamming.
#pragma once

#include <stdint.h>

#include "config.h"

namespace hey_jarvis {

class RecognizeCommands {
 public:
  RecognizeCommands();

  // Push the latest inference result.
  // ``positive_score`` is the uint8 probability of the wake-word class.
  // ``current_time_ms`` is millis().
  // Sets ``found_command`` to true when a fresh detection fires.
  void ProcessLatestResult(uint8_t positive_score, uint32_t current_time_ms,
                           bool* found_command, uint8_t* score_out);

 private:
  static constexpr int kMaxHistory = 32;

  struct Entry {
    uint32_t time_ms;
    uint8_t score;
  };

  Entry history_[kMaxHistory];
  int head_ = 0;
  int size_ = 0;
  uint32_t previous_top_label_ms_ = 0;
};

}  // namespace hey_jarvis
