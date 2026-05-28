// Audio provider — exposes a 1-second rolling buffer of int16 PCM samples
// captured from the Nano 33 BLE Sense onboard PDM microphone.
#pragma once

#include <stdint.h>

namespace hey_jarvis {

// Initialise the PDM peripheral.  Returns false on hardware failure.
bool AudioProviderBegin();

// Stop the PDM peripheral.  Safe to call from setup teardown.
void AudioProviderEnd();

// Returns the most-recently captured PCM window (kClipSamples samples).
// ``timestamp_ms`` is populated with the millis() when the buffer became
// available so the recogniser can compute time-decay.  Returns ``true``
// when ``out_samples`` was filled with fresh data, ``false`` otherwise.
bool AudioProviderGetSamples(int16_t* out_samples, uint32_t* timestamp_ms);

// Diagnostic counter (number of PDM ISR overruns).
uint32_t AudioProviderOverruns();

}  // namespace hey_jarvis
