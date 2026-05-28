#include "command_responder.h"

#include <Arduino.h>

#include "config.h"

namespace hey_jarvis {
namespace {

uint32_t g_led_off_at_ms = 0;
uint32_t g_total_detections = 0;

}  // namespace

void CommandResponderBegin() {
  pinMode(kStatusLedPin, OUTPUT);
  digitalWrite(kStatusLedPin, LOW);
}

void CommandResponderTick(uint32_t now_ms) {
  if (g_led_off_at_ms != 0 && now_ms >= g_led_off_at_ms) {
    digitalWrite(kStatusLedPin, LOW);
    g_led_off_at_ms = 0;
  }
}

void CommandResponderOnWakeWord(uint8_t score_q8, uint32_t now_ms) {
  ++g_total_detections;
  digitalWrite(kStatusLedPin, HIGH);
  g_led_off_at_ms = now_ms + kDetectionLedMs;

  const float confidence = score_q8 / 255.0f;
  Serial.println();
  Serial.println(F("Wake word detected!"));
  Serial.print(F("Confidence: "));
  Serial.print(static_cast<int>(confidence * 100.0f));
  Serial.println(F("%"));
  Serial.print(F("Total detections: "));
  Serial.println(g_total_detections);
  Serial.print(F("Uptime (ms): "));
  Serial.println(now_ms);
  Serial.println(F("--------------------"));
}

}  // namespace hey_jarvis
