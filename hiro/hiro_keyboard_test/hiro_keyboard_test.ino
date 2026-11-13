/*
 * HIRO keyboard movement test
 *
 * One step per key, so movement can be tuned before connecting EEG control.
 * Send characters from the Arduino Serial Monitor (115200 baud):
 *   w = forward step     s = backward step
 *   a = turn left (CCW)  d = turn right (CW)
 *   x = stand            + / - = faster / slower
 *   [ / ] = trim left / right (straightens the forward walk)
 * Characters typed while a step is running are discarded, so the robot
 * never "catches up" on a backlog of old commands.
 *
 * Requires the HIRO library (Library Manager -> "HIRO") and the
 * earlephilhower Raspberry Pi Pico board package.
 */

#include <HIRO.h>
#include <HIRO_Motion.h>

HIRO hiro;

int speed = 2;   // same value the official examples use
int trim = 0;    // walkForward/walkBackward trim

void setup() {
  hiro.begin();            // use hiro.genZeroBegin() for a Gen Zero build
  Serial.begin(115200);
  neutralStance(hiro);
  Serial.println("HIRO keyboard test: w/a/s/d move, x stand, +/- speed, [/] trim");
}

void loop() {
  if (!Serial.available()) return;
  char c = Serial.read();

  switch (c) {
    case 'w': walkForward(hiro, 0, 60, 25, speed, trim);  Serial.println("forward");  break;
    case 's': walkBackward(hiro, 0, 60, 25, speed, trim); Serial.println("backward"); break;
    case 'a': rotateCounterClockwise(hiro, 0, 60, 20, speed); Serial.println("left");  break;
    case 'd': rotateClockwise(hiro, 0, 60, 20, speed);        Serial.println("right"); break;
    case 'x': neutralStance(hiro); Serial.println("stand"); break;
    case '+': speed++;                 Serial.print("speed "); Serial.println(speed); break;
    case '-': speed = max(1, speed - 1); Serial.print("speed "); Serial.println(speed); break;
    case '[': trim--; Serial.print("trim "); Serial.println(trim); break;
    case ']': trim++; Serial.print("trim "); Serial.println(trim); break;
    default: return;  // ignore newlines and other characters
  }

  while (Serial.available()) Serial.read();  // drop anything sent during the move
}
