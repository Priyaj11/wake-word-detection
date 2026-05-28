#include "recognize_commands.h"

namespace hey_jarvis {

RecognizeCommands::RecognizeCommands() = default;

void RecognizeCommands::ProcessLatestResult(uint8_t positive_score,
                                            uint32_t current_time_ms,
                                            bool* found_command,
                                            uint8_t* score_out) {
  *found_command = false;
  *score_out = positive_score;

  // Push.
  history_[head_] = {current_time_ms, positive_score};
  head_ = (head_ + 1) % kMaxHistory;
  if (size_ < kMaxHistory) ++size_;

  // Sliding-window average across recent entries.
  const uint32_t window_start = (current_time_ms > kAveragingWindowMs)
                                    ? current_time_ms - kAveragingWindowMs
                                    : 0;
  uint32_t sum = 0;
  int count = 0;
  for (int i = 0; i < size_; ++i) {
    const Entry& e = history_[i];
    if (e.time_ms >= window_start) {
      sum += e.score;
      ++count;
    }
  }
  if (count < kMinDetectionsInWindow) return;
  const uint8_t average = static_cast<uint8_t>(sum / count);
  *score_out = average;

  if (average < kDetectionThresholdQ8) return;

  // Suppression: ignore detections too close to the previous one.
  if (current_time_ms - previous_top_label_ms_ < kSuppressionMs &&
      previous_top_label_ms_ != 0) {
    return;
  }
  previous_top_label_ms_ = current_time_ms;
  *found_command = true;
}

}  // namespace hey_jarvis
