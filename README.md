# Turtao Pi — Surveillance Robot Backend

Raspberry Pi backend for the Turtao surveillance robot. Camera capture, face
recognition, person tracking, anti-spoofing, ESP32 serial link, Bluetooth audio,
TTS alerts, and patrol loop — all served via Flask REST + WebSocket API.

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                         main.py                                   │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌─────────────────┐  │
│  │  Flask   │  │  WebSocket│  │   GUI    │  │    TurtaoCore   │  │
│  │  API     │  │  Status   │  │  (--gui) │  │  (orchestrator) │  │
│  └────┬─────┘  └────┬──────┘  └──────────┘  └────────┬────────┘  │
│       │              │                                │           │
├───────┴──────────────┴────────────────────────────────┴───────────┤
│                      Shared AppState                               │
│  ┌──────────┬──────────┬──────────┬──────────┬──────────────────┐  │
│  │ Sensor   │  Mode    │  Frame   │  Threat  │  Event Queue     │  │
│  │ Data     │  (IDLE/  │  Queue   │  State   │  (deque maxlen50)│  │
│  │          │  GUARD/  │ (deq 2)  │          │                  │  │
│  │          │  PATROL) │          │          │                  │  │
│  └──────────┴──────────┴──────────┴──────────┴──────────────────┘  │
├────────────────────────────────────────────────────────────────────┤
│  Camera    Face Rec   Tracker   Serial     BT/TTS    Patrol       │
│  Loop      Engine     (YOLO)    Link       Audio     Loop         │
│  (daemon)  (daemon)   (daemon)  (daemon)   (daemon)  (daemon)     │
└────────────────────────────────────────────────────────────────────┘
```

8 daemon threads: camera capture, face recognition, person tracker, serial
link, Bluetooth manager, wake word, patrol loop, WebSocket broadcast.

## Quick Start

```bash
# 1. Install (system deps + Python venv + models)
./install.sh

# 2. Activate
source venv/bin/activate

# 3. Copy and edit config
cp .env.example .env
# Edit .env with your ESP32 port, JBL MAC, etc.

# 4. Run tests
pytest

# 5. Run headless (production)
python main.py

# 6. Run with GUI (development — requires display)
python main.py --gui
```

## Environment Variables

| Variable       | Default      | Description            |
|----------------|-------------|------------------------|
| `FLASK_PORT`   | `5000`       | HTTP API port          |
| `ESP32_PORT`   | `/dev/ttyUSB0` | Serial device for ESP32 |
| `CAMERA_INDEX` | `0`          | Camera device index    |
| `JBL_MAC`      | —            | Bluetooth MAC of JBL   |

## JBL Bluetooth Pairing

```bash
# Scan
bluetoothctl scan on

# Pair
bluetoothctl pair AA:BB:CC:DD:EE:FF

# Trust (auto-reconnect)
bluetoothctl trust AA:BB:CC:DD:EE:FF

# Set in .env
echo "JBL_MAC=AA:BB:CC:DD:EE:FF" >> .env
```

## API Endpoints

| Method | Path                         | Description                |
|--------|------------------------------|----------------------------|
| GET    | `/api/mode`                  | Current operating mode     |
| POST   | `/api/mode`                  | Set mode (IDLE/GUARD/PATROL)|
| POST   | `/api/move`                  | Tank drive (ml, mr)        |
| POST   | `/api/control`               | Speed, nerf, safe, pan/tilt|
| GET    | `/api/environment`           | Sensor data                |
| GET    | `/api/battery`               | Battery status             |
| GET    | `/api/faces`                 | Enrolled faces list        |
| GET    | `/api/faces/unknowns`        | Unknown faces list          |
| POST   | `/api/faces/enroll/start`    | Start enrollment           |
| POST   | `/api/faces/enroll/capture`  | Capture pose               |
| POST   | `/api/faces/enroll/cancel`   | Cancel enrollment          |
| POST   | `/api/faces/promote`         | Promote unknown to enrolled|
| DELETE | `/api/faces/<name>`          | Delete enrolled face       |
| DELETE | `/api/faces/unknowns/<id>`   | Delete unknown face        |
| GET    | `/api/settings`              | Get all settings           |
| POST   | `/api/settings`              | Update settings            |
| POST   | `/api/settings/test-voice`   | Test TTS voice             |
| GET    | `/api/camera/stream`         | MJPEG stream               |
| POST   | `/api/alert`                 | Trigger alert              |
| GET    | `/api/ble/status`            | Bluetooth device status    |
| GET    | `/api/status`                | Combined robot status      |
| POST   | `/api/estop`                 | Emergency stop             |
| WS     | `/ws/status`                 | Real-time status push      |
| WS     | `/ws/intercom`               | Two-way audio              |

## Resolved Ambiguities

| Concern | Decision | Rationale |
|---------|----------|-----------|
| Thread safety | `AppState` uses `threading.Lock` with context manager | Lock protects all reads/writes; 8 threads + Flask + GUI all share one state object |
| Frame ownership | Camera loop writes `latest_frame` + appends to `frame_queue` (maxlen 2) | Readers get a reference; lock ensures no torn reads. Queue depth 2 prevents memory growth |
| GUI imports | `from turtao.gui.app_window import AppWindow` only in `--gui` branch | Zero `tkinter` imports in headless mode preserves headless compatibility |
| Model paths | Relative to project root (`models/`, `face_data/`, `piper/`) | Symlinks not required; `BASE_DIR` is `Path(__file__).parent.parent` |
| Enrollment flow | 5 poses × 8 frames = 40 captures total | Median embedding per pose stored; HOG face detection |
| Unknown faces | Saved as JPEG to `face_data/unknowns/` at 2s intervals | Dedup via filename timestamp; no DB needed |
| Settings persistence | `settings.json` at project root | Pydantic model with `shallow_merge` for partial updates |

## Testing

```bash
# Run all tests
pytest

# With coverage
pytest --cov=turtao --cov-report=term-missing

# Concurrency tests (stress-tests threading)
pytest tests/concurrency/

# Lint
ruff check turtao/

# Type check
mypy turtao/

# Type check (excluding GUI — tkinter not available on CI)
mypy turtao/ --exclude 'turtao/gui/'

# All checks
ruff check turtao/ && mypy turtao/ --exclude 'turtao/gui/' && pytest --cov=turtao
```

## Service Management

```bash
# Install systemd service (done by install.sh)
sudo systemctl enable turtao
sudo systemctl start turtao

# View logs
sudo journalctl -u turtao -f

# Restart
sudo systemctl restart turtao
```

## Project Layout

```
turtao-pi/
├── main.py                  # Entry point
├── install.sh               # Idempotent install script
├── pyproject.toml           # Project metadata, lint, type check config
├── requirements.txt         # Runtime dependencies
├── requirements-dev.txt     # Dev dependencies (pytest, ruff, mypy)
├── settings.json            # Persistent settings
├── .env                     # Environment variables (not in git)
├── .env.example             # Environment template
├── turtao/
│   ├── __init__.py
│   ├── core.py              # TurtaoCore orchestrator
│   ├── state.py             # AppState (thread-safe shared state)
│   ├── config.py            # AppConfig, Settings (Pydantic)
│   ├── logging_config.py    # Rotating file + console logging
│   ├── api/                 # Flask REST + WebSocket
│   │   ├── app_factory.py
│   │   ├── errors.py
│   │   ├── routes_*.py
│   │   ├── schemas.py
│   │   ├── ws_status.py
│   │   └── ws_intercom.py
│   ├── gui/                 # Development GUI (--gui only)
│   │   ├── app_window.py
│   │   ├── tab_enroll.py
│   │   ├── tab_faces.py
│   │   ├── tab_unknowns.py
│   │   ├── tab_sensors.py
│   │   ├── tab_settings.py
│   │   └── tab_log.py
│   ├── serial_link/         # ESP32 serial protocol
│   ├── vision/              # Camera, face rec, tracker, antispoof
│   ├── audio/               # TTS, Bluetooth, intercom
│   ├── hardware/            # Interfaces and mocks
│   ├── battery/             # Battery management
│   └── patrol/              # Patrol loop
├── tests/
│   ├── conftest.py          # Shared fixtures
│   ├── unit/
│   ├── integration/
│   └── concurrency/
├── face_data/               # Profiles, embeddings, unknowns
│   ├── profiles.json
│   ├── embeddings/
│   └── unknowns/
├── models/                  # YOLOv8n ONNX model
├── piper/                   # Piper TTS binary + voice model
├── sounds/                  # alert.wav (also BluetoothManager's connected-chime)
└── systemd/
    └── turtao.service       # Systemd unit file
```
