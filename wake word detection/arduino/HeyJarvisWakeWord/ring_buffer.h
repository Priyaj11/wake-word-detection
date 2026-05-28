// Tiny lock-light single-producer / single-consumer ring buffer for int16
// PCM samples from the PDM ISR to the inference loop.
#pragma once

#include <stdint.h>
#include <string.h>

namespace hey_jarvis {

template <int Capacity>
class RingBuffer {
 public:
  RingBuffer() : head_(0), tail_(0) {}

  // Producer side — typically called from the PDM ISR.
  // Returns the number of samples actually written.
  int Write(const int16_t* samples, int count) {
    int written = 0;
    for (int i = 0; i < count; ++i) {
      const int next = (head_ + 1) % Capacity;
      if (next == tail_) {  // full — drop sample
        break;
      }
      buffer_[head_] = samples[i];
      head_ = next;
      ++written;
    }
    return written;
  }

  // Consumer side — copy at most ``count`` samples without removing them.
  int Peek(int16_t* out, int count) const {
    int available = Available();
    if (count > available) count = available;
    int t = tail_;
    for (int i = 0; i < count; ++i) {
      out[i] = buffer_[t];
      t = (t + 1) % Capacity;
    }
    return count;
  }

  // Consumer side — drop ``count`` samples from the tail.
  void Drop(int count) {
    int available = Available();
    if (count > available) count = available;
    tail_ = (tail_ + count) % Capacity;
  }

  int Available() const {
    int h = head_, t = tail_;
    if (h >= t) return h - t;
    return Capacity - (t - h);
  }

  void Clear() { head_ = tail_ = 0; }

 private:
  int16_t buffer_[Capacity];
  volatile int head_;
  volatile int tail_;
};

}  // namespace hey_jarvis
