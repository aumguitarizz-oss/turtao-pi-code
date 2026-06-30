#!/bin/bash
# install.sh — Turtao full setup for Raspberry Pi 5 (64-bit Bookworm)
# Safe to re-run — skips anything already installed.
# Usage: bash install.sh

set -e

TURTAO_DIR="$HOME/turtao"
VENV="$TURTAO_DIR/venv"
PIP="$VENV/bin/pip"
PYTHON="$VENV/bin/python3"

echo "========================================="
echo "  Turtao Install — Raspberry Pi 5"
echo "========================================="

pkg_installed() { dpkg -s "$1" &>/dev/null; }
pip_installed()  { "$PIP" show "$1" &>/dev/null 2>&1; }
skip()           { echo "  [OK] $1 already present"; }

# ── [1/9] system update ───────────────────────────────────────────────────────
echo ""
echo "[1/9] Updating package index..."
sudo apt-get update -qq

# ── [2/9] system packages ─────────────────────────────────────────────────────
echo ""
echo "[2/9] Checking system dependencies..."

SYSTEM_PKGS=(
    python3-pip python3-venv python3-dev
    cmake build-essential
    libboost-all-dev
    libopenblas-dev liblapack-dev
    libavcodec-dev libavformat-dev libswscale-dev
    libgtk-3-dev
    libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev
    libatlas-base-dev gfortran
    libjpeg-dev libpng-dev libtiff-dev
    portaudio19-dev python3-pyaudio
    python3-tk
    bluetooth bluez
    pulseaudio pulseaudio-module-bluetooth
    pavucontrol
    i2c-tools
    git wget curl
    ffmpeg
    v4l-utils
)

TO_INSTALL=()
for pkg in "${SYSTEM_PKGS[@]}"; do
    if pkg_installed "$pkg"; then
        echo "  [OK] $pkg"
    else
        echo "  [--] $pkg will install"
        TO_INSTALL+=("$pkg")
    fi
done

if [ ${#TO_INSTALL[@]} -gt 0 ]; then
    echo "  Installing ${#TO_INSTALL[@]} missing package(s)..."
    sudo apt-get install -y "${TO_INSTALL[@]}"
else
    echo "  All system packages already installed."
fi

# ── [3/9] hardware interfaces ─────────────────────────────────────────────────
echo ""
echo "[3/9] Checking hardware interfaces..."

if [ "$(raspi-config nonint get_i2c 2>/dev/null)" = "0" ]; then
    skip "I2C"
else
    echo "  Enabling I2C..."
    sudo raspi-config nonint do_i2c 0
fi

if [ "$(raspi-config nonint get_serial_hw 2>/dev/null)" = "0" ]; then
    skip "Serial hardware"
else
    echo "  Enabling serial hardware..."
    sudo raspi-config nonint do_serial_hw 0
    sudo raspi-config nonint do_serial_cons 1
fi

for grp in dialout audio bluetooth i2c video; do
    if id -nG "$USER" | grep -qw "$grp"; then
        echo "  [OK] group: $grp"
    else
        echo "  Adding $USER to group: $grp"
        sudo usermod -aG "$grp" "$USER"
    fi
done

# ── [4/9] python venv ─────────────────────────────────────────────────────────
echo ""
echo "[4/9] Checking Python virtual environment..."

if [ -f "$VENV/bin/activate" ]; then
    skip "venv at $VENV"
else
    echo "  Creating venv..."
    cd "$TURTAO_DIR"
    python3 -m venv venv
fi

source "$VENV/bin/activate"
pip install --upgrade pip setuptools wheel -q

# ── [5/9] python packages ─────────────────────────────────────────────────────
echo ""
echo "[5/9] Checking Python packages..."

# Install in dependency-safe order. face_recognition (dlib) last — slow compile.
PY_PKGS=(
    numpy
    pillow
    scipy
    opencv-python
    flask
    flask-socketio
    pyserial
    smbus2
    adafruit-circuitpython-ina219
    python-dotenv
    pybluez2
    pyaudio
    openai-whisper
    openwakeword
    face_recognition
)

for pkg in "${PY_PKGS[@]}"; do
    if pip_installed "$pkg"; then
        echo "  [OK] $pkg"
    else
        if [ "$pkg" = "face_recognition" ]; then
            echo ""
            echo "  [!!] Installing face_recognition..."
            echo "       dlib compiles from source: 20-30 min on Pi 5."
            echo "       Do NOT interrupt this step."
            echo ""
        else
            echo "  Installing $pkg..."
        fi
        pip install "$pkg" -q
        echo "  [OK] $pkg installed"
    fi
done
echo "  All Python packages ready."

# ── [6/9] piper tts ───────────────────────────────────────────────────────────
echo ""
echo "[6/9] Checking Piper TTS..."

mkdir -p "$TURTAO_DIR/piper"

if [ -x "$TURTAO_DIR/piper/piper" ]; then
    skip "piper binary"
else
    echo "  Downloading Piper binary (ARM64)..."
    cd "$TURTAO_DIR/piper"
    wget -q --show-progress \
        https://github.com/rhasspy/piper/releases/download/2023.11.14-2/piper_linux_aarch64.tar.gz
    tar -xzf piper_linux_aarch64.tar.gz
    rm  piper_linux_aarch64.tar.gz
    chmod +x piper
    echo "  [OK] Piper binary ready."
fi

if [ -f "$TURTAO_DIR/piper/en_US-amy-medium.onnx" ]; then
    skip "Amy voice model"
else
    echo "  Downloading Amy voice model (~60 MB)..."
    cd "$TURTAO_DIR/piper"
    BASE="https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/amy/medium"
    wget -q --show-progress "${BASE}/en_US-amy-medium.onnx"
    wget -q --show-progress "${BASE}/en_US-amy-medium.onnx.json"
    echo "  [OK] Voice model ready."
fi

# ── [7/9] openwakeword models ─────────────────────────────────────────────────
echo ""
echo "[7/9] Checking openWakeWord models..."

OWW_DIR="$HOME/.local/share/openwakeword"
if [ -d "$OWW_DIR" ] && [ "$(ls -A "$OWW_DIR" 2>/dev/null)" ]; then
    skip "openWakeWord models at $OWW_DIR"
else
    echo "  Downloading openWakeWord models..."
    "$PYTHON" -c "
import openwakeword
openwakeword.utils.download_models()
print('  [OK] openWakeWord models ready.')
"
fi

# ── [8/9] audio assets ────────────────────────────────────────────────────────
echo ""
echo "[8/9] Checking audio assets..."

mkdir -p "$TURTAO_DIR/sounds"

if [ -f "$TURTAO_DIR/sounds/silence.wav" ]; then
    skip "sounds/silence.wav"
else
    echo "  Generating silence.wav (JBL Go 3 keepalive)..."
    "$PYTHON" - << 'PY'
import wave, struct, os
path = os.path.expanduser("~/turtao/sounds/silence.wav")
with wave.open(path, "w") as f:
    f.setnchannels(1); f.setsampwidth(2); f.setframerate(44100)
    f.writeframes(struct.pack("<" + "h" * 44100 * 30, *([0] * 44100 * 30)))
print("  [OK] silence.wav created")
PY
fi

if [ -f "$TURTAO_DIR/sounds/alert.wav" ]; then
    skip "sounds/alert.wav"
else
    echo "  Generating alert.wav (880 Hz)..."
    "$PYTHON" - << 'PY'
import wave, struct, math, os
path = os.path.expanduser("~/turtao/sounds/alert.wav")
rate, freq = 44100, 880
samples = [int(32767 * math.sin(2 * math.pi * freq * i / rate))
           for i in range(rate)]
with wave.open(path, "w") as f:
    f.setnchannels(1); f.setsampwidth(2); f.setframerate(rate)
    f.writeframes(struct.pack("<" + "h" * len(samples), *samples))
print("  [OK] alert.wav created")
PY
fi

# ── [9/9] systemd service ─────────────────────────────────────────────────────
echo ""
echo "[9/9] Checking systemd service..."

SERVICE_SRC="$TURTAO_DIR/systemd/turtao.service"
SERVICE_DST="/etc/systemd/system/turtao.service"

if [ ! -f "$SERVICE_SRC" ]; then
    echo "  [!!] systemd/turtao.service not found -- skipping."
elif [ -f "$SERVICE_DST" ] && cmp -s "$SERVICE_SRC" "$SERVICE_DST"; then
    skip "turtao.service (unchanged)"
else
    echo "  Installing/updating turtao.service..."
    sudo cp "$SERVICE_SRC" "$SERVICE_DST"
    sudo systemctl daemon-reload
    echo "  [OK] systemd updated."
fi

if [ -f "$SERVICE_DST" ]; then
    if systemctl is-enabled turtao.service &>/dev/null; then
        echo "  [OK] turtao.service already enabled"
    else
        echo "  Enabling turtao.service..."
        sudo systemctl enable turtao.service
    fi
fi

# ── .env ──────────────────────────────────────────────────────────────────────
echo ""
if [ ! -f "$TURTAO_DIR/.env" ]; then
    cp "$TURTAO_DIR/.env.example" "$TURTAO_DIR/.env"
    echo "[!!] Created .env from template."
    echo "     Edit ~/turtao/.env -- set JBL_MAC before starting."
else
    echo "[OK] .env already exists"
fi

echo ""
echo "========================================="
echo "  Installation complete!"
echo "========================================="
echo ""
echo "  Production (headless / systemd):"
echo "    sudo reboot  # if added to new groups"
echo "    sudo systemctl start turtao.service"
echo "    journalctl -u turtao -f"
echo ""
echo "  Dev GUI:"
echo "    cd ~/turtao"
echo "    source venv/bin/activate"
echo "    python3 gui.py"
echo ""
