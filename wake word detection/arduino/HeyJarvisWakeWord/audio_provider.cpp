// Implementation of audio_provider.h using the official PDM Arduino
// library shipped with the Nano 33 BLE Sense core.
#include "audio_provider.h"

#include <Arduino.h>
#include <PDM.h>

#include "config.h"
#include "ring_buffer.h"

namespace hey_jarvis {
namespace {

// 4× the clip length keeps a safe margin between the ISR and the loop.
constexpr int kRingCapacity = kClipSamples * 4;
RingBuffer<kRingCapacity> g_ring;

volatile uint32_t g_overruns = 0;

// Scratch buffer used inside the PDM ISR.
constexpr int kPdmBufferBytes = 512;
int16_t g_pdm_buffer[kPdmBufferBytes / 2];

void OnPdmData() {
  const int bytes_available = PDM.available();
  if (bytes_available <= 0) return;
  const int read = PDM.read(g_pdm_buffer, bytes_available);
  const int samples = read / 2;
  const int written = g_ring.Write(g_pdm_buffer, samples);
  if (written != samples) {
    ++g_overruns;
  }
}

}  // namespace

bool AudioProviderBegin() {
  PDM.onReceive(OnPdmData);
  PDM.setBufferSize(kPdmBufferBytes);
  // Optional: PDM.setGain(40);  // adjust 0–80 if needed
  if (!PDM.begin(1 /*channels*/, kAudioSampleFrequency)) {
    return false;
  }
  return true;
}

void AudioProviderEnd() {
  PDM.end();
  g_ring.Clear();
}

bool AudioProviderGetSamples(int16_t* out_samples, uint32_t* timestamp_ms) {
  if (g_ring.Available() < kClipSamples) return false;
  // Slide window forward by 25 % of the clip to overlap inferences.
  const int stride = kClipSamples / 4;
  int peeked = g_ring.Peek(out_samples, kClipSamples);
  if (peeked < kClipSamples) return false;
  g_ring.Drop(stride);
  if (timestamp_ms) *timestamp_ms = millis();
  return true;
}

uint32_t AudioProviderOverruns() { return g_overruns; }

}  // namespace hey_jarvis
