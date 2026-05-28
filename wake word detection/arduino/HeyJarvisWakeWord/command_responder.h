// Reacts to a wake-word detection by turning on the onboard LED and
// printing a structured message to Serial.
#pragma once

#include <stdint.h>

namespace hey_jarvis {

void CommandResponderBegin();
void CommandResponderTick(uint32_t now_ms);
void CommandResponderOnWakeWord(uint8_t score_q8, uint32_t now_ms);

}  // namespace hey_jarvis
