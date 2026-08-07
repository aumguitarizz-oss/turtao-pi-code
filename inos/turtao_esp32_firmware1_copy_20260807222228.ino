#include <Wire.h>
#include <ArduinoJson.h>
#include <Adafruit_VL53L0X.h>
#include <DHT.h>

// ================================================================
// Turtao — ESP32-S3 Firmware (stripped build, single ToF, no pan/tilt)
// Handles: MQ-2 + DHT22 sensors, 1x VL53L0X time-of-flight (front
// bumper), JSON protocol with the Pi, forwarding motor/estop commands
// to ATX2+ over UART, and an autonomous wall-avoidance maneuver:
// forward -> wall detected -> reverse -> turn right -> resume forward.
// Camera is a fixed webcam on this build -- no servo gimbal, no
// "servo" command.
//
// Pin map:
//   GPIO1  MQ-2 analog   (ADC1_CH0, via 10k/20k divider)
//   GPIO5  DHT22         (exterior, 1-wire)
//   GPIO8  I2C SDA       (VL53L0X direct connect)
//   GPIO9  I2C SCL
//   GPIO17 UART2 TX -> level shifter -> ATX2+ RXD1 (Port 2)
//   GPIO18 UART2 RX <- level shifter <- ATX2+ TXD1 (Port 3)
//
// MOTOR POLARITY (critical): the ATX2+ motor channels are inverted on
// this build — a positive M command physically drives the wheels
// BACKWARD. Every motor write goes through driveRaw(), which takes
// RAW command values, so forward motion is sent as a negative value.
// ================================================================

// ---------- Pin map ----------
#define PIN_SDA        8
#define PIN_SCL        9
#define PIN_MQ2        1   // ADC1_CH0
#define PIN_DHT        5   // single DHT22 (exterior)
#define UART_TX_TO_ATX2    17
#define UART_RX_FROM_ATX2  18

// VL53L0X reports very large distances (commonly ~8191mm) when it
// sees no valid target. Anything at or above this is treated as
// "out of range" -> null, same as a hard sensor failure.
#define TOF_OUT_OF_RANGE_MM 8000

// ---------- Wall-avoidance (bumper) tunables ----------
#define BUMPER_THRESHOLD_MM 300   // any front ToF below this triggers the maneuver
#define BUMPER_SAMPLE_MS    100   // how often the front ToF is polled
#define BUMPER_REVERSE_MS   600   // how long the reverse phase runs
#define BUMPER_TURN_MS      500   // how long the right-turn phase runs
#define BUMPER_POWER        50    // maneuver motor magnitude (raw, 0-100)
#define BUMPER_CRUISE_POWER 40    // forward speed used if no move command was seen

// Motor clamp — enforced here regardless of what the Pi sends.
const float MOTOR_CLAMP = 0.8;

DHT dht(PIN_DHT, DHT22);
Adafruit_VL53L0X lox = Adafruit_VL53L0X();
const char* tofName = "tof_front";
bool tofSensorOk = false;

// ---------- Bumper state ----------
enum BumperPhase { BUMPER_IDLE, BUMPER_REVERSE, BUMPER_TURN_RIGHT };
BumperPhase bumperPhase = BUMPER_IDLE;
unsigned long bumperPhaseStart = 0;
unsigned long lastBumperSample = 0;

// Last move command from the Pi, in desired (forward-positive) units.
float lastMl = 0.0;
float lastMr = 0.0;

// ---------- Read the ToF sensor, applying the out-of-range fix ----------
// Returns true if a real in-range reading was obtained (value written
// to outDistanceMm). Returns false if out of range or sensor absent.
bool readTof(int &outDistanceMm) {
  if (!tofSensorOk) return false;

  VL53L0X_RangingMeasurementData_t measure;
  lox.rangingTest(&measure, false);

  if (measure.RangeStatus == 4) return false;  // library's own "invalid" status
  if (measure.RangeMilliMeter >= TOF_OUT_OF_RANGE_MM) return false;  // sentinel value fix

  outDistanceMm = measure.RangeMilliMeter;
  return true;
}

// ---------- Raw motor write to ATX2+ ----------
// leftRaw/rightRaw are RAW command values: positive = physical backward
// on this build (see polarity note at top). Left side = channels 1,3;
// right side = channels 2,4.
void driveRaw(int leftRaw, int rightRaw) {
  Serial1.print("M,1,"); Serial1.println(leftRaw);
  Serial1.print("M,3,"); Serial1.println(leftRaw);
  Serial1.print("M,2,"); Serial1.println(rightRaw);
  Serial1.print("M,4,"); Serial1.println(rightRaw);
}

// ---------- Build and send the sensor payload ----------
void sendSensorPayload() {
  StaticJsonDocument<256> doc;

  int distance;
  if (readTof(distance)) {
    doc[tofName] = distance;
  } else {
    doc[tofName] = nullptr;
  }

  int mq2Raw = analogRead(PIN_MQ2);
  doc["gas_mq2"] = (mq2Raw / 4095.0) * 3.3;

  float t = dht.readTemperature();
  float h = dht.readHumidity();
  if (isnan(t)) {
    doc["temp_dht"] = nullptr;
  } else {
    doc["temp_dht"] = t;
  }

  if (isnan(h)) {
    doc["humidity"] = nullptr;
  } else {
    doc["humidity"] = h;
  }

  serializeJson(doc, Serial);
  Serial.println();
}

// ---------- Wall-avoidance state machine ----------
// Non-blocking, driven from loop(). Only armed while a forward move is
// commanded (lastMl/lastMr both > 0). Runs reverse -> turn right, then
// resumes forward at the last commanded speed.
void bumperTick() {
  unsigned long now = millis();

  if (bumperPhase == BUMPER_IDLE) {
    // Sample the front ToF on a timer; only react when moving forward.
    if (now - lastBumperSample < BUMPER_SAMPLE_MS) return;
    lastBumperSample = now;
    if (lastMl <= 0 || lastMr <= 0) return;

    int frontDist;
    if (!readTof(frontDist)) return; 
    if (frontDist >= BUMPER_THRESHOLD_MM) return;

    // Wall ahead: reverse out. Positive raw = physical backward.
    bumperPhase = BUMPER_REVERSE;
    bumperPhaseStart = now;
    driveRaw(BUMPER_POWER, BUMPER_POWER);
    return;
  }

  if (bumperPhase == BUMPER_REVERSE) {
    if (now - bumperPhaseStart < BUMPER_REVERSE_MS) return;
    // Right pivot: left forward (negative raw), right backward (positive).
    bumperPhase = BUMPER_TURN_RIGHT;
    bumperPhaseStart = now;
    driveRaw(-BUMPER_POWER, BUMPER_POWER);
    return;
  }

  if (bumperPhase == BUMPER_TURN_RIGHT) {
    if (now - bumperPhaseStart < BUMPER_TURN_MS) return;
    // Resume forward at the last commanded speed, or default cruise.
    bumperPhase = BUMPER_IDLE;
    float ml = lastMl > 0 ? lastMl : BUMPER_CRUISE_POWER / 100.0;
    float mr = lastMr > 0 ? lastMr : BUMPER_CRUISE_POWER / 100.0;
    int leftRaw = -(int)(ml * 100);
    int rightRaw = -(int)(mr * 100);
    driveRaw(leftRaw, rightRaw);
  }
}

// ---------- "move" command: clamp, negate polarity, forward to ATX2+ ----------
void handleMove(JsonDocument &cmdDoc) {
  float ml = cmdDoc["ml"] | 0.0;
  float mr = cmdDoc["mr"] | 0.0;

  if (ml > MOTOR_CLAMP) ml = MOTOR_CLAMP;
  if (ml < -MOTOR_CLAMP) ml = -MOTOR_CLAMP;
  if (mr > MOTOR_CLAMP) mr = MOTOR_CLAMP;
  if (mr < -MOTOR_CLAMP) mr = -MOTOR_CLAMP;

  // Remember the commanded direction; the bumper resumes from this.
  lastMl = ml;
  lastMr = mr;

  // While a bumper maneuver is running, record the command but don't
  // fight the maneuver — it gets applied when the maneuver completes.
  if (bumperPhase != BUMPER_IDLE) return;

  // Polarity: ATX2 channels are inverted, so +ml/+mr (desired forward)
  // must be negated before hitting the motors.
  driveRaw(-(int)(ml * 100), -(int)(mr * 100));

  StaticJsonDocument<64> ack;
  ack["status"] = "ok";
  ack["ml"] = ml;
  ack["mr"] = mr;
  serializeJson(ack, Serial);
  Serial.println();
}

// ---------- "estop" command ----------
void handleEstop() {
  bumperPhase = BUMPER_IDLE;
  lastMl = 0.0;
  lastMr = 0.0;
  Serial1.println("X");
  StaticJsonDocument<32> ack;
  ack["status"] = "stopped";
  serializeJson(ack, Serial);
  Serial.println();
}

void setup() {
  Serial.begin(115200);          // to Pi, native USB
  Serial1.begin(9600, SERIAL_8N1, UART_RX_FROM_ATX2, UART_TX_TO_ATX2);  // to ATX2+
  delay(500);

  Wire.begin(PIN_SDA, PIN_SCL);
  dht.begin();
  
  // Initialize the single ToF sensor
  tofSensorOk = lox.begin();
}

void loop() {
  bumperTick();

  if (Serial.available()) {
    String line = Serial.readStringUntil('\n');
    line.trim();
    if (line.length() == 0) return;

    StaticJsonDocument<256> cmdDoc;
    DeserializationError err = deserializeJson(cmdDoc, line);
    if (err) return;  // malformed JSON -> ignore silently, never crash

    const char* cmd = cmdDoc["cmd"] | "";

    if (strcmp(cmd, "sensors") == 0) {
      sendSensorPayload();
    } else if (strcmp(cmd, "move") == 0) {
      handleMove(cmdDoc);
    } else if (strcmp(cmd, "estop") == 0) {
      handleEstop();
    }
    // unrecognized cmd -> ignore silently
  }
}