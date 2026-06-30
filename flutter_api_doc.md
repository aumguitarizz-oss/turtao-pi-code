# Turtao Flutter App — AI Build Guide

This document tells you everything you need to build the Turtao Flutter app correctly.
Read it fully before writing any code.

---

## 1. What Turtao Is

Turtao is a home surveillance robot running on a Raspberry Pi 5.
The Pi runs a Python Flask server (port 5000) that the Flutter app connects to over Tailscale VPN.

The app is the owner's remote interface. It shows the live camera feed, alerts when an intruder is detected, lets the owner switch modes, manage enrolled faces, and view sensor data.

---

## 2. Network Architecture

```
Phone (Flutter app)
    |
    | Tailscale VPN  (no port forwarding needed, no public IP)
    |
Raspberry Pi 5
    - Flask REST API on port 5000
    - SocketIO WebSocket on port 5000 (same server)
    - MJPEG stream on port 5000
```

The user enters the Pi's Tailscale IP (e.g. `100.x.x.x`) or Tailscale hostname in the app settings.
Base URL: `http://<tailscale-ip>:5000`
No authentication. Tailscale handles security.

---

## 3. Flutter Tech Stack

```yaml
dependencies:
  flutter:
    sdk: flutter
  dio: ^5.4.0                    # HTTP client
  socket_io_client: ^2.0.3+1    # WebSocket / SocketIO
  riverpod: ^2.5.1               # state management
  flutter_riverpod: ^2.5.1
  shared_preferences: ^2.2.3     # store Pi IP, settings cache
  cached_network_image: ^3.3.1   # face thumbnails
  http: ^1.2.1                   # for MJPEG chunked stream
```

**Do not use** `video_player` for the camera stream — it does not support MJPEG.
You need a custom MJPEG widget (see Section 8).

---

## 4. App Screens

| Screen | Purpose |
|--------|---------|
| Connection Setup | Enter Pi Tailscale IP, test connection |
| Home | Live camera, threat badge, mode buttons, hw status, battery |
| Faces | List enrolled faces, delete, start enrollment |
| Enrollment Flow | Step-by-step 5-pose face capture |
| Unknowns | Grid of unknown detections, promote to named face |
| Sensors | All ESP32 sensor readings |
| Settings | Tolerance, speed, TTS toggles, stealth mode, JBL MAC |
| Event Log | Scrolling live event stream from robot |

---

## 5. State Machine

The robot is always in one of these states:

```
IDLE    → recognition off, motors stopped, LED off
GUARD   → recognition active, motors stopped, LED green
PATROL  → recognition active, motors autonomous, LED blue blink
THREAT  → overlaid on GUARD or PATROL when unknown face detected
```

The app must reflect these states clearly. On THREAT:
- Show a full-width red alert banner with the name and confidence
- Flash or pulse the banner while threat is active
- Clear automatically when threat resolves (next SocketIO status update)

---

## 6. REST API Reference

### GET /api/health
System status check. Use for connection test on startup.

**Response:**
```json
{
  "status": "ok",
  "mode": "IDLE",
  "phase": 1,
  "uptime_seconds": 3600,
  "hw": {
    "camera": true,
    "esp32": true,
    "ipst": true,
    "battery": true,
    "bluetooth": false
  },
  "room": "living_room",
  "timestamp": "2026-06-28T10:00:00+00:00"
}
```

---

### GET /api/alert
Current threat state. Poll this if SocketIO is unavailable.

**Response:**
```json
{
  "threat": true,
  "confidence": 0.42,
  "state": "THREAT",
  "name": "Unknown",
  "timestamp": "2026-06-28T10:00:01+00:00"
}
```

`state` is one of: `"IDLE"`, `"GUARD"`, `"PATROL"`, `"THREAT"`

---

### GET /api/stream
MJPEG live video stream. Do NOT use with normal HTTP GET — see Section 8 for implementation.

Returns `multipart/x-mixed-replace; boundary=frame`.
Frame rate: ~30 fps. JPEG quality: 70%.
Resolution: 640×480 (annotated with face boxes and labels).

---

### GET /api/battery
Battery state from INA219 (on ESP32 I2C bus, data relayed via serial JSON).

**Response:**
```json
{
  "voltage": 7.82,
  "current_ma": 420.0,
  "percentage": 74,
  "charging": false,
  "danger": false
}
```

`danger: true` means voltage < 6.6V. Show a prominent warning.
`voltage: 0.0` means no battery data received from ESP32 yet.

---

### GET /api/environment
All sensor readings from ESP32.

**Response:**
```json
{
  "temp_dht": 28.5,
  "humid": 65.2,
  "temp_bmp": 28.1,
  "pressure": 1013.2,
  "gas_mq2": 120,
  "gas_mq135": 180,
  "sound": 45,
  "pir": false,
  "accel_x": 0.02,
  "accel_y": -0.01,
  "accel_z": 9.81,
  "gyro_x": 0.0,
  "gyro_y": 0.0,
  "gyro_z": 0.0,
  "tof_fl": 450,
  "tof_fc": 380,
  "tof_fr": 420,
  "tof_down": 12,
  "battery_voltage": 7.82,
  "battery_current_ma": 420.0,
  "battery_pct": 74,
  "battery_charging": false,
  "ble_phone_present": true,
  "ulp_wake": false,
  "wifi_rssi": {
    "HomeWiFi": -42,
    "NeighborWiFi": -78
  }
}
```

ToF values in mm. 9999 = sensor timeout / object out of range.

---

### GET /api/hw
Hardware connection flags only.

**Response:**
```json
{
  "camera": true,
  "esp32": true,
  "ipst": false,
  "battery": true,
  "bluetooth": true
}
```

---

### GET /api/room
Current room estimate and phone BLE presence.

**Response:**
```json
{
  "room": "living_room",
  "phone_present": true
}
```

---

### GET /api/faces
List of enrolled face profiles.

**Response:**
```json
[
  {
    "name": "Alex",
    "thumb_url": "/api/faces/Alex/thumb"
  },
  {
    "name": "Jordan",
    "thumb_url": "/api/faces/Jordan/thumb"
  }
]
```

---

### GET /api/faces/{name}/thumb
Returns a JPEG image of the enrolled face (first reference photo).
Use with `Image.network()` or `CachedNetworkImage`.

---

### GET /api/faces/unknowns
List of auto-saved unknown intruder crops.

**Response:**
```json
[
  {
    "id": "1717000000000",
    "timestamp": "2026-06-28T10:05:00+00:00",
    "image_url": "/api/faces/unknowns/1717000000000"
  }
]
```

---

### GET /api/faces/unknowns/{uid}
Returns a JPEG crop of the unknown face.

---

### GET /api/faces/enroll/status
Returns current enrollment session state. Returns 404 if no session active.

**Response:**
```json
{
  "name": "Alex",
  "pose": 2,
  "total": 5
}
```

---

### POST /api/faces/enroll/start
Begin a 5-pose face enrollment session.

**Request:**
```json
{ "name": "Alex" }
```

**Response:**
```json
{
  "name": "Alex",
  "pose": 0,
  "total": 5,
  "pose_name": "Face directly forward"
}
```

---

### POST /api/faces/enroll/capture
Capture the current pose. The robot samples up to 30 camera frames, applies quality checks (blur, brightness, face size), and saves the average embedding.

**Request:** empty body

**Response (success):**
```json
{
  "success": true,
  "complete": false,
  "pose": 1,
  "pose_name": "Turn slightly left (~30°)"
}
```

**Response (all 5 poses done):**
```json
{
  "success": true,
  "complete": true
}
```

**Response (quality failure — retry same pose):**
```json
{
  "success": false,
  "reason": "blur/size/light",
  "retry": true
}
```

**Important:** Capture takes ~1.5 seconds (30 frames × 50ms each). Show a spinner. Do not allow the user to tap Capture again until the response arrives.

---

### POST /api/faces/promote
Promote an unknown face crop to a named enrolled profile.

**Request:**
```json
{
  "unknown_id": "1717000000000",
  "name": "Jordan"
}
```

**Response:**
```json
{ "ok": true }
```

**Error responses:**
```json
{ "error": "not found" }
{ "error": "no face found in image" }
```

---

### DELETE /api/faces/{name}
Delete all embeddings and reference images for a named face.

**Response:**
```json
{ "ok": true }
```

---

### POST /api/mode
Switch robot mode.

**Request:**
```json
{ "mode": "PATROL" }
```

Valid values: `"IDLE"`, `"GUARD"`, `"PATROL"`

**Response:**
```json
{ "ok": true, "mode": "PATROL" }
```

---

### POST /api/move
Direct motor command. Only meaningful in IDLE or GUARD (in PATROL the patrol loop takes over).

**Request:**
```json
{ "ml": 0.5, "mr": 0.5 }
```

`ml` = left motor, `mr` = right motor. Range: -0.8 to +0.8. Hard cap enforced server-side.

Positive = forward. Negative = reverse.
Differential steering: `ml=0.5, mr=-0.5` = turn right on the spot.

**Response:**
```json
{ "ok": true }
```

---

### GET /api/settings
Current settings dict.

**Response:**
```json
{
  "tts_enabled": true,
  "tts_threat": true,
  "tts_gas": true,
  "tts_tamper": true,
  "tts_patrol": true,
  "tts_wake": true,
  "speed": 0.8,
  "safe_mode": false,
  "tolerance": 0.52,
  "stealth_mode": false,
  "brightness_threshold": 60,
  "jbl_mac": "AA:BB:CC:DD:EE:FF",
  "strobe_on_threat": true,
  "ble_auto_disarm": true,
  "room_fingerprints": {}
}
```

---

### POST /api/settings
Update one or more settings. Merges into existing settings — only send keys you want to change.

**Request:**
```json
{ "tolerance": 0.60, "stealth_mode": true }
```

**Response:**
```json
{ "ok": true }
```

---

### POST /api/led
Control the 12V truck LEDs (via IRLZ44N MOSFETs on ESP32 GPIO).

**Request:**
```json
{ "led": "white", "mode": "on", "pwm": 200 }
```

- `led`: `"white"` or `"red"`
- `mode`: `"on"`, `"off"`, `"strobe"`
- `pwm`: 0–255 brightness (only relevant for `mode: "on"`)

---

### POST /api/strobe
Trigger the flashbang strobe deterrent.

**Request:**
```json
{ "duration_ms": 3000 }
```

**Response:**
```json
{ "ok": true }
```

---

### POST /api/room/calibrate
Save current WiFi RSSI as the fingerprint for a named room.
The robot must be physically in the room when this is called.

**Request:**
```json
{ "room": "living_room" }
```

**Response:**
```json
{
  "ok": true,
  "room": "living_room",
  "rssi": { "HomeWiFi": -42, "NeighborWiFi": -78 }
}
```

**Error if no wifi_rssi data yet:**
```json
{ "error": "no wifi_rssi data from ESP32" }
```

---

## 7. WebSocket (SocketIO) Events

The robot broadcasts a `status` event every 2 seconds to all connected clients.

**Connection:**
```dart
import 'package:socket_io_client/socket_io_client.dart' as IO;

final socket = IO.io(
  'http://<tailscale-ip>:5000',
  IO.OptionBuilder()
    .setTransports(['websocket'])
    .disableAutoConnect()
    .build(),
);

socket.connect();
socket.on('status', (data) {
  // handle status update
});
```

**Event payload:**
```json
{
  "mode": "GUARD",
  "threat": false,
  "confidence": 0.0,
  "battery_pct": 74,
  "hw": {
    "camera": true,
    "esp32": true,
    "ipst": false,
    "battery": true,
    "bluetooth": true
  },
  "room": "living_room",
  "phone": true,
  "connected": true,
  "timestamp": "2026-06-28T10:00:05+00:00"
}
```

**Important:** The socket payload does not include sensor readings (that would be too large at 2s interval). Use GET /api/environment for sensor data, polled every 2–5 seconds on the Sensors screen.

---

## 8. MJPEG Stream Implementation

MJPEG is a series of JPEG frames separated by `--frame\r\n` boundaries.
Flutter's `Image.network` does not support it. You need to parse the chunked HTTP response manually.

```dart
import 'dart:async';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;

class MjpegStream extends StatefulWidget {
  final String url;
  const MjpegStream({super.key, required this.url});

  @override
  State<MjpegStream> createState() => _MjpegStreamState();
}

class _MjpegStreamState extends State<MjpegStream> {
  Uint8List? _frameBytes;
  StreamSubscription? _sub;
  final List<int> _buffer = [];

  static final _boundary = '--frame\r\n'.codeUnits;
  static final _jpegStart = [0xFF, 0xD8];
  static final _jpegEnd   = [0xFF, 0xD9];

  @override
  void initState() {
    super.initState();
    _startStream();
  }

  void _startStream() async {
    try {
      final client   = http.Client();
      final request  = http.Request('GET', Uri.parse(widget.url));
      final response = await client.send(request);
      _sub = response.stream.listen(
        (chunk) {
          _buffer.addAll(chunk);
          _extractFrames();
        },
        onError: (_) => Future.delayed(
          const Duration(seconds: 2), _startStream),
        onDone: () => Future.delayed(
          const Duration(seconds: 2), _startStream),
      );
    } catch (_) {
      await Future.delayed(const Duration(seconds: 2));
      _startStream();
    }
  }

  void _extractFrames() {
    while (true) {
      // find JPEG start marker
      final start = _indexOfSeq(_buffer, _jpegStart, 0);
      if (start == -1) break;
      // find JPEG end marker after start
      final end = _indexOfSeq(_buffer, _jpegEnd, start);
      if (end == -1) break;
      final frameEnd = end + 2;
      final frameBytes = Uint8List.fromList(
        _buffer.sublist(start, frameEnd));
      _buffer.removeRange(0, frameEnd);
      if (mounted) {
        setState(() => _frameBytes = frameBytes);
      }
    }
    // prevent buffer growing unbounded
    if (_buffer.length > 500000) {
      _buffer.clear();
    }
  }

  int _indexOfSeq(List<int> haystack, List<int> needle, int from) {
    outer:
    for (int i = from; i <= haystack.length - needle.length; i++) {
      for (int j = 0; j < needle.length; j++) {
        if (haystack[i + j] != needle[j]) continue outer;
      }
      return i;
    }
    return -1;
  }

  @override
  void dispose() {
    _sub?.cancel();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    if (_frameBytes == null) {
      return const AspectRatio(
        aspectRatio: 640 / 480,
        child: ColoredBox(
          color: Colors.black,
          child: Center(child: CircularProgressIndicator()),
        ),
      );
    }
    return AspectRatio(
      aspectRatio: 640 / 480,
      child: Image.memory(
        _frameBytes!,
        gaplessPlayback: true,
        fit: BoxFit.contain,
      ),
    );
  }
}
```

Use it like this on the Home screen:

```dart
MjpegStream(url: 'http://$piIp:5000/api/stream')
```

---

## 9. Enrollment Flow (Step by Step)

The enrollment flow is a multi-step process. The app drives it.

```
1. User enters a name
2. POST /api/faces/enroll/start  {"name": "Alex"}
   → response: {pose: 0, pose_name: "Face directly forward", total: 5}
3. Show instruction: "Face directly forward"
4. User taps Capture
5. Show spinner / "Capturing..." — disable button
6. POST /api/faces/enroll/capture  (empty body)
   Wait up to 5 seconds (capture takes ~1.5s + server processing)
7a. Response {success: true, complete: false, pose: 1, pose_name: "..."}
    → update progress, show next instruction, re-enable Capture
7b. Response {success: false, retry: true, reason: "blur/size/light"}
    → show error "Poor image quality. Try better lighting or move closer."
    → re-enable Capture for retry — do NOT advance pose counter
7c. Response {success: true, complete: true}
    → enrollment done — show success, navigate back to Faces list
8. Repeat from step 4 for each of the 5 poses
```

**UI requirements:**
- Show a progress indicator (e.g. 5 dots, filled as poses complete)
- Display the pose instruction text clearly
- Disable the Capture button while a request is in flight
- Handle timeout: if request takes > 8 seconds, show error and allow retry
- Allow Cancel at any step (just navigate away — no server-side cancel needed)

---

## 10. Connection Management

```dart
class ConnectionState {
  final String baseUrl;        // "http://100.x.x.x:5000"
  final bool connected;        // /api/health returned ok
  final DateTime? lastSeen;
}
```

On app startup:
1. Load saved IP from SharedPreferences
2. GET /api/health
3. If success → mark connected, start SocketIO
4. If failure → show connection screen

On SocketIO disconnect:
- Try reconnect every 3 seconds
- Show connection status indicator on all screens

Connection status indicator: a small coloured dot in the app bar.
Green = connected. Grey = disconnected. Show last known state when grey.

---

## 11. Error Handling

All Dio requests should use a try/catch with:
- `DioException.connectionTimeout` → show "Cannot reach robot"
- `DioException.connectionError` → show "Robot offline"
- Non-2xx status → show the `error` field from response JSON
- Timeout: set Dio `connectTimeout` and `receiveTimeout` to 5000ms

Enrollment capture is the only endpoint with long processing time (~1.5s).
Set a separate timeout of 8000ms for that call.

---

## 12. Key Implementation Notes

### Riverpod providers needed

```dart
// Current connection config
@riverpod
class ConnectionConfig extends _$ConnectionConfig {...}

// Live robot status (from SocketIO)
@riverpod
class RobotStatus extends _$RobotStatus {...}

// Faces list
@riverpod
Future<List<FaceProfile>> facesList(FacesListRef ref) async {...}

// Sensor data (polled)
@riverpod
Future<SensorData> sensorData(SensorDataRef ref) async {...}

// Settings
@riverpod
Future<Settings> robotSettings(RobotSettingsRef ref) async {...}
```

### Settings screen

Use `GET /api/settings` to load, and `POST /api/settings` on any change.
Send only the changed key — not the whole settings object — to avoid accidentally overwriting other settings changed on the Pi side.

```dart
await dio.post('/api/settings', data: {'tolerance': 0.60});
```

### Mode buttons

Show the current mode visually (highlighted button).
Optimistically update the UI on tap, then confirm with SocketIO status event.

### Threat notification

When `threat: true` arrives in a SocketIO status event AND the app is in the background, trigger a local notification using `flutter_local_notifications`.

### Face thumbnail URLs

```dart
Image.network(
  'http://$piIp:5000/api/faces/$name/thumb',
  errorBuilder: (_, __, ___) => const Icon(Icons.person),
)
```

Thumbnails are only available if enrollment completed successfully.

### Battery display

- 0% data (`voltage == 0`) → show "-- " (no data from ESP32 yet)
- `danger: true` → show battery icon in red with warning
- `charging: true` → show charging indicator

---

## 13. Screens Detail

### Home Screen
- Top bar: connection dot, battery %, current mode
- Full-width MJPEG camera (16:9 crop or letterbox)
- Threat banner (hidden when no threat, red full-width when active)
- Three mode buttons: IDLE / GUARD / PATROL
- Bottom bar: hw status pills (CAM, ESP32, IPST, BATT, BT)

### Sensors Screen
- Sections: Environment, IMU, ToF, Battery, BLE/WiFi
- All values from `GET /api/environment`, polled every 3 seconds
- ToF values: show a simple diagram of the robot with distances
- Gas values: show a warning colour if > threshold (MQ2 > 300, MQ135 > 400)

### Faces Screen
- Scrollable grid of enrolled face thumbnails with names
- Long press to delete (with confirm dialog)
- FAB to start enrollment

### Enrollment Screen
- Step-by-step, driven by the enrollment flow in Section 9
- 5 pose steps shown as a progress row at top
- Large instruction text in the middle
- Capture button at bottom
- Camera preview (same MJPEG widget, smaller) so user can see themselves

### Unknowns Screen
- List of unknown crops with timestamps
- Tap to enlarge
- "Promote" button → shows a name input dialog → calls POST /api/faces/promote

### Settings Screen
- Grouped by category: TTS, Movement, Security, Hardware
- Sliders for tolerance (0.3–0.9) and speed (0.1–0.8)
- Toggle switches for all boolean settings
- Text field for JBL MAC address
- "Connection" section at top: current Pi IP, change button

### Log Screen
- Scrolling text list
- Polled from `GET /api/health` → ... actually logs are NOT exposed via API.
- Implement as: display local SocketIO event history (store all status events locally)
- OR: skip this screen for MVP

---

## 14. API Endpoints Not Relevant to Flutter

These are used internally or by the tkinter dev GUI:
- `POST /api/room/calibrate` — optional advanced feature
- `POST /api/led` — optional manual control
- `POST /api/strobe` — can be triggered from Settings screen as a test

---

## 15. Data Models (Dart)

```dart
class RobotStatus {
  final String mode;          // IDLE, GUARD, PATROL
  final bool threat;
  final double confidence;
  final int batteryPct;
  final HwStatus hw;
  final String room;
  final bool phonePresent;
  final DateTime timestamp;
}

class HwStatus {
  final bool camera;
  final bool esp32;
  final bool ipst;
  final bool battery;
  final bool bluetooth;
}

class FaceProfile {
  final String name;
  final String thumbUrl;
}

class UnknownFace {
  final String id;
  final DateTime timestamp;
  final String imageUrl;
}

class BatteryInfo {
  final double voltage;
  final double currentMa;
  final int percentage;
  final bool charging;
  final bool danger;
}

class SensorData {
  final double tempDht;
  final double humid;
  final double tempBmp;
  final double pressure;
  final int gasMq2;
  final int gasMq135;
  final int sound;
  final bool pir;
  final double accelX, accelY, accelZ;
  final double gyroX, gyroY, gyroZ;
  final int tofFl, tofFc, tofFr, tofDown;
  final double batteryVoltage;
  final int batteryPct;
  final bool batteryCharging;
  final bool blePhonePresent;
}
```

---

## 16. Quick Start Checklist for Building the App

1. Create Flutter project: `flutter create turtao_app`
2. Add all dependencies from Section 3
3. Build `ConnectionConfigProvider` — reads/saves IP from SharedPreferences
4. Build `MjpegStream` widget from Section 8 exactly
5. Set up SocketIO listener → update `RobotStatus` Riverpod state
6. Build Home screen with MJPEG, threat banner, mode buttons
7. Build Faces screen with `GET /api/faces`, thumbnails, delete
8. Build Enrollment flow following Section 9 step by step
9. Build Unknowns screen with promote flow
10. Build Sensors screen with 3-second polling
11. Build Settings screen with per-key PATCH pattern
12. Add local notifications for background threat alerts
13. Test all screens with Pi connected over Tailscale
