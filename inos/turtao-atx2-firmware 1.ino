#include <ATX2.h>

// ================================================================
// Turtao — ATX2+ Firmware
// Executes motor/servo commands forwarded from the ESP32-S3.
//
// Protocol (comma-delimited, newline-terminated, over Port 2 RXD1 /
// Port 3 TXD1, 9600 baud — must match the ESP32-S3 sketch):
//   M,<channel>,<power>     e.g. M,1,70   -> motor(1, 70)
//   S,<channel>,<angle>     e.g. S,1,90   -> servo(1, 90)
//   X                       -> emergency stop, all motors to 0
//
// ASSUMPTION STILL TO VERIFY: this uses the standard Arduino Serial1
// object for Port 2/Port 3. If ATX2.h exposes this UART through its
// own custom functions instead, swap the Serial1.* calls below for
// whatever ATX2.h actually provides.
// ================================================================

#define CMD_BUFFER_SIZE 32
char cmdBuffer[CMD_BUFFER_SIZE];
uint8_t bufIndex = 0;

void setup() {
  Serial1.begin(9600);   // must match ESP32-S3's Serial1.begin() baud rate

  glcdClear();
  setTextSize(2);
  glcd(1, 0, "Turtao ATX2+");
  glcd(3, 0, "Cmd listener");
  glcd(5, 0, "Waiting...");
}

// ---------- Parse and execute one complete command line ----------
void executeCommand(char* line) {
  int len = strlen(line);
  while (len > 0 && (line[len - 1] == '\r' || line[len - 1] == '\n' || line[len - 1] == ' ')) {
    line[--len] = '\0';
  }
  if (len == 0) return;

  if (line[0] == 'X') {
    for (int ch = 1; ch <= 4; ch++) motor_stop(ch);
    glcdClear();
    glcd(1, 0, "ESTOP");
    return;
  }

  char* type = strtok(line, ",");
  char* chStr = strtok(NULL, ",");
  char* valStr = strtok(NULL, ",");

  if (type == NULL || chStr == NULL || valStr == NULL) {
    return;  // malformed line -> ignore silently
  }

  int channel = atoi(chStr);
  int value = atoi(valStr);

  if (type[0] == 'M') {
    if (channel < 1 || channel > 6) return;
    if (value > 100) value = 100;
    if (value < -100) value = -100;
    motor(channel, value);

    glcdClear();
    glcd(1, 0, "Motor %d", channel);
    glcd(3, 0, "%d%%", value);

  } else if (type[0] == 'S') {
    if (channel < 1 || channel > 8) return;
    if (value < 0) value = 0;
    if (value > 180) value = 180;
    servo(channel, value);

    glcdClear();
    glcd(1, 0, "Servo %d", channel);
    glcd(3, 0, "%d deg", value);
  }
  // unrecognized type -> ignore silently
}

void loop() {
  while (Serial1.available()) {
    char c = Serial1.read();

    if (c == '\n') {
      cmdBuffer[bufIndex] = '\0';
      executeCommand(cmdBuffer);
      bufIndex = 0;
    } else if (bufIndex < CMD_BUFFER_SIZE - 1) {
      cmdBuffer[bufIndex++] = c;
    } else {
      bufIndex = 0;  // overflow protection -> reset rather than corrupt
    }
  }
}
