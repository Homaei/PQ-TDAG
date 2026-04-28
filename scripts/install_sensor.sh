#!/bin/bash
# scripts/install_sensor.sh
#
# One-time setup for ICS sensor hardware:
#   - Raspberry Pi 5  (Cortex-A76, Ubuntu 24.04 aarch64)
#   - NVIDIA Jetson Orin Nano  (Cortex-A78AE + GA10B GPU, JetPack 6.x)
#
# The script auto-detects which platform it is running on and
# adjusts accordingly. On Jetson, it additionally checks for
# CUDA availability (JetPack ships nvcc by default).
#
# Usage:
#   chmod +x scripts/install_sensor.sh
#   bash scripts/install_sensor.sh

set -e
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'
log()  { echo -e "${GREEN}[ok]${NC} $1"; }
warn() { echo -e "${YELLOW}[!!]${NC} $1"; }
info() { echo -e "${CYAN}[--]${NC} $1"; }

echo ""
echo "======================================================"
echo "  PQ-TDAG Sensor Setup"
echo "======================================================"

# ── Detect platform ───────────────────────────────────────────
PLATFORM="unknown"
if [ -f /proc/device-tree/model ]; then
    MODEL=$(cat /proc/device-tree/model 2>/dev/null || echo "")
    if echo "$MODEL" | grep -qi "raspberry"; then
        PLATFORM="rpi5"
    elif echo "$MODEL" | grep -qi "jetson\|orin"; then
        PLATFORM="jetson"
    fi
fi

# Fallback: check CPU
if [ "$PLATFORM" = "unknown" ]; then
    CPU=$(grep -m1 "Model name" /proc/cpuinfo 2>/dev/null | cut -d: -f2 | xargs)
    if echo "$CPU" | grep -qi "cortex-a76\|BCM2712"; then
        PLATFORM="rpi5"
    elif echo "$CPU" | grep -qi "cortex-a78\|carmel"; then
        PLATFORM="jetson"
    fi
fi

echo ""
info "Detected platform: $PLATFORM"
echo ""

# ── System packages ───────────────────────────────────────────
log "Installing system dependencies..."
apt-get update -qq
apt-get install -y \
    build-essential cmake git \
    python3-pip python3-venv python3-dev \
    libssl-dev libffi-dev \
    htop

# ── Python virtualenv ─────────────────────────────────────────
log "Creating Python virtual environment at ~/pq_tdag_env..."
python3 -m venv ~/pq_tdag_env
source ~/pq_tdag_env/bin/activate
pip install --upgrade pip setuptools wheel -q

# ── Python packages ───────────────────────────────────────────
log "Installing Python packages..."
pip install -q \
    liboqs-python \
    numpy scipy matplotlib \
    tqdm psutil pycryptodome

# ── Verify liboqs ─────────────────────────────────────────────
log "Verifying liboqs on ARM..."
python3 -c "
import oqs, platform
enabled = oqs.get_enabled_sig_mechanisms()
print(f'  liboqs {oqs.oqs_version()} on {platform.machine()}')
print(f'  {len(enabled)} signature schemes available')
# Report which ISA extensions are available on ARM
import subprocess
try:
    r = subprocess.run(['grep','-m1','Features','/proc/cpuinfo'],
                       capture_output=True, text=True)
    flags = r.stdout.split(':')[-1].strip()
    neon  = 'asimd' in flags or 'neon' in flags
    sha   = 'sha2' in flags
    aes   = 'aes' in flags
    print(f'  Neon/ASIMD: {neon}  SHA2: {sha}  AES: {aes}')
except:
    pass
"

# ── Jetson-specific: CUDA check ───────────────────────────────
if [ "$PLATFORM" = "jetson" ]; then
    echo ""
    log "Jetson platform detected — checking CUDA..."
    if command -v nvcc &>/dev/null; then
        nvcc --version | grep "release"
        log "CUDA available. GPU benchmark will compile automatically."
        info "Compile command:"
        info "  nvcc -O3 -arch=sm_87 src/gpu_batch_verify.cu -loqs -o gpu_bench_jetson"
    else
        warn "nvcc not found. JetPack should include it."
        warn "Check: dpkg -l | grep cuda"
    fi
fi

# ── Record hardware info ──────────────────────────────────────
mkdir -p results/logs
python3 -c "
import platform, json, subprocess
from datetime import datetime
try:
    import oqs; liboqs_ver = oqs.oqs_version()
except: liboqs_ver = 'unknown'
try:
    r = subprocess.run(['cat','/proc/device-tree/model'],
                       capture_output=True, text=True)
    model = r.stdout.strip()
except: model = platform.processor()
info = {
    'generated': datetime.now().isoformat(),
    'platform':  '$PLATFORM',
    'model':     model,
    'arch':      platform.machine(),
    'os':        platform.platform(),
    'python':    platform.python_version(),
    'liboqs':    liboqs_ver,
    'role':      'sensor',
}
with open('results/logs/hardware_info_sensor.json','w') as f:
    json.dump(info, f, indent=2)
print('  Hardware info saved.')
" 2>/dev/null || true

echo ""
echo "======================================================"
log "Sensor setup complete on: $PLATFORM"
echo ""
echo "  Activate environment:  source ~/pq_tdag_env/bin/activate"
if [ "$PLATFORM" = "rpi5" ]; then
    echo "  Run sensor benchmark:  python3 step_2c_rpi5_sensor_benchmark.py"
else
    echo "  Run sensor benchmark:  python3 step_2d_jetson_sensor_benchmark.py"
fi
echo "======================================================"
