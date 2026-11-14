/*
 * HIRO Wi-Fi control (Raspberry Pi Pico 2 W)
 *
 * Receives motion commands over UDP (port 5005) from the Vanguard decoder
 * (live.py --robot <ip>) or the keyboard tester (robot/keyboard_teleop.py).
 *
 * Two network modes, chosen automatically:
 *   - Access point (default, no setup): HIRO creates its own Wi-Fi network
 *     "HIRO-Vanguard". Join it from the laptop; HIRO is always 192.168.42.1.
 *   - Join a network: if secrets.h exists in this folder, HIRO joins that
 *     network instead and prints the address it was given.
 *
 * Accepted messages, one per UDP packet:
 *   JSON  {"command": "FORWARD", "confidence": 0.81, ...}   (what live.py sends)
 *   text  FORWARD | BACKWARD | LEFT | RIGHT | STOP, or w / s / a / d / x
 *
 * Behaviour:
 *   - one step per command; packets that arrive during a step are dropped
 *     except the most recent, so the robot never works through a backlog
 *   - safety stop: no valid message for WATCHDOG_MS -> stand still
 *   - Wi-Fi lost (join mode) -> stand still and reconnect
 *   - LED: red = no Wi-Fi, green = ready and idle, blue = moving
 *
 * Join mode setup: copy secrets.h.example to secrets.h and fill in the
 * network name and password (secrets.h is git-ignored).
 */

#include <WiFi.h>
#include <WiFiUdp.h>
#include <HIRO.h>
#include <HIRO_Motion.h>

#if __has_include("secrets.h")
#include "secrets.h"
#define JOIN_NETWORK 1
#else
#define JOIN_NETWORK 0
#endif

// ---- Access point settings (used when there is no secrets.h) ----
// Anyone who knows this password can drive the robot: change it for your team.
const char *AP_SSID = "HIRO-Vanguard";
const char *AP_PASSWORD = "vanguard2026";   // at least 8 characters

// ---- Tuning (use the values found with hiro_keyboard_test) ----
const int SPEED = 2;
const int TRIM = 0;
const unsigned long WATCHDOG_MS = 1000;   // stop if no command for this long
const float MIN_CONFIDENCE = 0.0;         // ignore moves below this decoder confidence
const unsigned int UDP_PORT = 5005;

HIRO hiro;
WiFiUDP udp;
char packet[512];

String command = "STOP";
unsigned long lastMessage = 0;
bool standing = false;
unsigned long lastIpPrint = 0;

void led(int r, int g, int b) {
  analogWrite(hiro.red, r);
  analogWrite(hiro.green, g);
  analogWrite(hiro.blue, b);
}

void stand() {
  if (!standing) {
    neutralStance(hiro);
    standing = true;
  }
}

IPAddress myIP() {
  return JOIN_NETWORK ? WiFi.localIP() : WiFi.softAPIP();
}

bool networkUp() {
  return JOIN_NETWORK ? WiFi.status() == WL_CONNECTED : true;
}

void startAccessPoint() {
  led(10, 0, 0);
  stand();
  Serial.print("Starting access point ");
  Serial.println(AP_SSID);
  WiFi.mode(WIFI_AP);
  while (!WiFi.softAP(AP_SSID, AP_PASSWORD)) {
    Serial.println("Access point failed to start, retrying...");
    delay(1000);
  }
  udp.begin(UDP_PORT);
  Serial.print("Ready. Join Wi-Fi \"");
  Serial.print(AP_SSID);
  Serial.print("\" and send commands to ");
  Serial.println(myIP());
  led(0, 10, 0);
}

#if JOIN_NETWORK
void connectWiFi() {
  led(10, 0, 0);
  stand();
  Serial.print("Connecting to ");
  Serial.println(WIFI_SSID);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  unsigned long start = millis();
  while (WiFi.status() != WL_CONNECTED) {
    delay(250);
    if (millis() - start > 20000) {          // retry every 20 s
      Serial.println("Still trying...");
      WiFi.disconnect();
      WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
      start = millis();
    }
  }
  udp.begin(UDP_PORT);
  Serial.print("Connected. HIRO IP address: ");
  Serial.println(WiFi.localIP());
  led(0, 10, 0);
}
#endif

void startNetwork() {
#if JOIN_NETWORK
  connectWiFi();
#else
  startAccessPoint();
#endif
}

// Extract a command from one packet. Returns "" if the packet is not usable.
String parseCommand(const char *text) {
  String msg(text);
  msg.trim();
  msg.toUpperCase();

  String cmd = msg;
  int key = msg.indexOf("\"COMMAND\"");
  if (key >= 0) {                                   // JSON message
    int q1 = msg.indexOf('"', msg.indexOf(':', key) + 1);
    int q2 = msg.indexOf('"', q1 + 1);
    if (q1 < 0 || q2 < 0) return "";
    cmd = msg.substring(q1 + 1, q2);

    int ck = msg.indexOf("\"CONFIDENCE\"");
    if (ck >= 0 && cmd != "STOP") {
      float conf = msg.substring(msg.indexOf(':', ck) + 1).toFloat();
      if (conf < MIN_CONFIDENCE) return "";
    }
  }

  if (cmd == "W") cmd = "FORWARD";
  else if (cmd == "S") cmd = "BACKWARD";
  else if (cmd == "A") cmd = "LEFT";
  else if (cmd == "D") cmd = "RIGHT";
  else if (cmd == "X") cmd = "STOP";

  if (cmd == "FORWARD" || cmd == "BACKWARD" || cmd == "LEFT" || cmd == "RIGHT" || cmd == "STOP") {
    return cmd;
  }
  return "";
}

// Read every queued packet; keep only the newest valid command.
void readPackets() {
  while (udp.parsePacket() > 0) {
    int len = udp.read(packet, sizeof(packet) - 1);
    if (len <= 0) continue;
    packet[len] = '\0';
    String cmd = parseCommand(packet);
    if (cmd.length() > 0) {
      if (cmd != command) {
        Serial.print("command: ");
        Serial.println(cmd);
      }
      command = cmd;
      lastMessage = millis();
    }
  }
}

void setup() {
  hiro.begin();
  Serial.begin(115200);
  stand();
  delay(1500);  // time to open the Serial Monitor
  Serial.println("HIRO Wi-Fi control");
  startNetwork();
}

void loop() {
  if (!networkUp()) {
    command = "STOP";
    startNetwork();
  }

  readPackets();

  if (command != "STOP" && millis() - lastMessage > WATCHDOG_MS) {
    Serial.println("safety stop: no commands received");
    command = "STOP";
  }

  if (millis() - lastIpPrint > 10000) {   // reprint so a late-opened monitor still sees it
    Serial.print("HIRO IP address: ");
    Serial.println(myIP());
    lastIpPrint = millis();
  }

  if (command == "STOP") {
    led(0, 10, 0);
    stand();
    delay(5);
    return;
  }

  led(0, 0, 10);
  standing = false;
  if (command == "FORWARD")       walkForward(hiro, 0, 60, 25, SPEED, TRIM);
  else if (command == "BACKWARD") walkBackward(hiro, 0, 60, 25, SPEED, TRIM);
  else if (command == "LEFT")     rotateCounterClockwise(hiro, 0, 60, 20, SPEED);
  else if (command == "RIGHT")    rotateClockwise(hiro, 0, 60, 20, SPEED);
}
