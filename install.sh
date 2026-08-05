#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$REPO_DIR/venv"
MODELS_DIR="$REPO_DIR/models"
PIPER_DIR="$REPO_DIR/piper"
SOUNDS_DIR="$REPO_DIR/sounds"
FACE_DATA_DIR="$REPO_DIR/face_data"
SYSTEMD_DIR="$REPO_DIR/systemd"

echo "=== Turtao Pi Installer ==="

# ── System packages ──────────────────────────────────────────────────
echo ">>> Installing system packages..."
sudo apt-get update -qq
sudo apt-get install -y -qq \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    git \
    cmake \
    build-essential \
    libssl-dev \
    libffi-dev \
    libhdf5-dev \
    libhdf5-dev \
    portaudio19-dev \
    libsdl2-dev \
    libsdl2-mixer-dev \
    libsdl2-image-dev \
    libsdl2-ttf-dev \
    flac \
    bluetooth \
    bluez \
    bluez-tools \
    pulseaudio-module-bluetooth \
    pi-bluetooth \
    alsa-utils

# ── Serial enable & user groups ──────────────────────────────────────
echo ">>> Enabling serial + user groups..."
sudo raspi-config nonint do_serial 0 2>/dev/null || true
sudo usermod -a -G dialout "$USER"
sudo usermod -a -G bluetooth "$USER"
sudo usermod -a -G audio "$USER"

# ── Python venv ──────────────────────────────────────────────────────
echo ">>> Creating Python virtual environment..."
if [ ! -d "$VENV_DIR" ]; then
    # Using pyenv Python 3.12 because mediapipe and others lack 3.13 wheels
    "$HOME/.pyenv/versions/3.12.8/bin/python" -m venv "$VENV_DIR"
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
pip install --upgrade pip setuptools wheel --quiet

# ── Python packages (in order) ──────────────────────────────────────
echo ">>> Installing Python packages..."
pip install --quiet -r "$REPO_DIR/requirements.txt"

# ── Dev packages ────────────────────────────────────────────────────
echo ">>> Installing dev packages..."
pip install --quiet \
    pytest==7.4.0 \
    pytest-cov==4.1.0 \
    ruff==0.1.0 \
    mypy==1.7.0

# ── YOLOv8n ONNX export ─────────────────────────────────────────────
echo ">>> Exporting YOLOv8n ONNX model..."
mkdir -p "$MODELS_DIR"
if [ ! -f "$MODELS_DIR/yolov8n.onnx" ]; then
    python3 -c "
import torch
from ultralytics import YOLO
model = YOLO('yolov8n.pt')
model.export(format='onnx', imgsz=640)
import shutil
shutil.move('yolov8n.onnx', '$MODELS_DIR/yolov8n.onnx')
"
fi

# ── MediaPipe pose landmarker ──────────────────────────────────────
if [ ! -f "$MODELS_DIR/pose_landmarker_lite.task" ]; then
    echo ">>> Downloading MediaPipe pose landmarker model..."
    wget -q "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task" \
        -O "$MODELS_DIR/pose_landmarker_lite.task"
fi

# ── Piper TTS download (ARM64) ──────────────────────────────────────
echo ">>> Downloading Piper TTS (ARM64)..."
mkdir -p "$PIPER_DIR"
if [ ! -f "$PIPER_DIR/piper" ]; then
    PIPER_VER="2023.11.14-2"
    ARCH="linux_aarch64"
    wget -q "https://github.com/rhasspy/piper/releases/download/$PIPER_VER/piper_$ARCH.tar.gz" \
        -O /tmp/piper.tar.gz
    tar -xzf /tmp/piper.tar.gz -C "$PIPER_DIR"
    rm /tmp/piper.tar.gz
fi
# Download voice model
if [ ! -f "$PIPER_DIR/en_US-amy-medium.onnx" ]; then
    wget -q "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium/en_US-amy-medium.onnx" \
        -O "$PIPER_DIR/en_US-amy-medium.onnx"
    wget -q "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium/en_US-amy-medium.onnx.json" \
        -O "$PIPER_DIR/en_US-amy-medium.onnx.json"
fi

# ── openWakeWord models ─────────────────────────────────────────────
echo ">>> Downloading openWakeWord models..."
python3 -c "
import openwakeword
openwakeword.utils.download_models()
" 2>/dev/null || echo "(openWakeWord models may need manual download)"

# ── silence.wav and alert.wav ──────────────────────────────────────
echo ">>> Generating audio files..."
mkdir -p "$SOUNDS_DIR"
if [ ! -f "$SOUNDS_DIR/silence.wav" ]; then
    python3 -c "
import wave
import struct
import math
sr = 22050
dur = 1
with wave.open('$SOUNDS_DIR/silence.wav', 'w') as w:
    w.setnchannels(1)
    w.setsampwidth(2)
    w.setframerate(sr)
    w.writeframes(b'\x00\x00' * sr * dur)
"
fi
if [ ! -f "$SOUNDS_DIR/alert.wav" ]; then
    python3 -c "
import wave
import struct
import math
sr = 22050
dur = 0.5
freq = 880
with wave.open('$SOUNDS_DIR/alert.wav', 'w') as w:
    w.setnchannels(1)
    w.setsampwidth(2)
    w.setframerate(sr)
    samples = []
    for i in range(int(sr * dur)):
        val = int(16000 * math.sin(2 * math.pi * freq * i / sr))
        samples.append(val)
    w.writeframes(struct.pack('<' + 'h' * len(samples), *samples))
"
fi

# ── Face data directories ──────────────────────────────────────────
echo ">>> Creating face data directories..."
mkdir -p "$FACE_DATA_DIR/embeddings"
mkdir -p "$FACE_DATA_DIR/unknowns"
if [ ! -f "$FACE_DATA_DIR/profiles.json" ]; then
    echo "[]" > "$FACE_DATA_DIR/profiles.json"
fi

# ── Systemd service ─────────────────────────────────────────────────
echo ">>> Installing systemd service..."
sudo cp "$SYSTEMD_DIR/turtao.service" /etc/systemd/system/turtao.service
sudo systemctl daemon-reload

# ── Done ────────────────────────────────────────────────────────────
echo ""
echo "=== Install complete ==="
echo ""
echo "Next steps:"
echo "  1. cp .env.example .env  # edit with your settings"
echo "  2. source venv/bin/activate"
echo "  3. pytest                 # verify installation"
echo "  4. python main.py         # run headless"
echo "  5. python main.py --gui   # run with dev GUI (requires display)"
echo ""
echo "Systemd service installed: sudo systemctl enable turtao"
