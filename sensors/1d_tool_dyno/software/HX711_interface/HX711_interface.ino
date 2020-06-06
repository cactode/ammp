#include <HX711.h>

#define LOADCELL_DOUT_PIN 4
#define LOADCELL_SCK_PIN 5

HX711 loadcell;

void setup() {
  Serial.begin(115200);
  loadcell.begin(LOADCELL_DOUT_PIN, LOADCELL_SCK_PIN);
  loadcell.tare();
}

void loop() {
  if (Serial.available() > 0) {
    char inByte = Serial.read();
    if (inByte == 'F') {
      float weight = -loadcell.get_value();
      Serial.println(weight);
    } else if (inByte == 'T') {
      loadcell.tare();
      Serial.println("TARED");
    }
  }
}
