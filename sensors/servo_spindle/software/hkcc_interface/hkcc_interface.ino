#include <due_can.h>

#define CAN_COMM_MB_IDX 0
#define CAN_TRANSFER_ID 0x07
#define CAN_TX_PRIO 0

CAN_FRAME pack_frame(float p, float v, float t) {
  
}

void setup() {
  // put your setup code here, to run once:
  Serial.begin(115200);
  Can0.begin(CAN_BPS_1000K);
  Can0.watchFor();
}

void loop() {
  // put your main code here, to run repeatedly:
  
}
