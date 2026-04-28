#!/bin/bash
# scripts/install_gateway.sh
#
# One-time setup for the edge gateway machine (x86, Ubuntu 22.04+).
# This installs liboqs, creates the Python environment, and configures
# the CPU for reproducible timing measurements.
#
# Usage:
#   chmod +x scripts/install_gateway.sh
#   bash scripts/install_gateway.sh
#
# After completion:
#   source ~/pq_tdag_env/bin/activate

set -e
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
log()  { echo -e "${GREEN}[ok]${NC} $1"; }
warn() { echo -e "${YELLOW}[!!]${NC} $1"; }

echo ""
echo "======================================================"
echo "  PQ-TDAG Gateway Setup"
echo "======================================================"
echo ""

# ── System packages ───────────────────────────────────────────
log "Installing system dependencies..."
apt-get update -qq
apt-get install -y \
    build-essential cmake git curl wget \
    python3-pip python3-venv python3-dev \
    libssl-dev libffi-dev \
    cpufrequtils linux-tools-generic \
    htop nvtop 2>/dev/null || true

# ── Python virtualenv ─────────────────────────────────────────
log "Creating Python virtual environment at ~/pq_tdag_env..."
python3 -m venv ~/pq_tdag_env
source ~/pq_tdag_env/bin/activate
pip install --upgrade pip setuptools wheel -q

# ── Python packages ───────────────────────────────────────────
log "Installing Python packages..."
pip install -q \
    liboqs-python \
    numpy scipy matplotlib seaborn pandas \
    tqdm psutil tabulate \
    cryptography pycryptodome

# ── Verify liboqs ─────────────────────────────────────────────
log "Verifying liboqs installation..."
python3 -c "
import oqs
enabled = oqs.get_enabled_sig_mechanisms()
print(f'  liboqs {oqs.oqs_version()} — {len(enabled)} schemes available')
for s in ['ML-DSA-44','ML-DSA-65','Falcon-512']:
    status = 'ok' if s in enabled else 'MISSING'
    print(f'  [{status}] {s}')
" || { warn "liboqs verification failed. Try: pip install liboqs-python --force-reinstall"; }

# ── CPU performance mode ──────────────────────────────────────
log "Setting CPU governor to performance mode..."
if command -v cpupower &>/dev/null; then
    cpupower frequency-set -g performance 2>/dev/null && \
        log "Governor set to performance." || \
        warn "Could not set governor (try: sudo cpupower ...)"
else
    warn "cpupower not found. Install: apt install linux-tools-$(uname -r)"
    warn "Timing measurements may have higher variance without performance governor."
fi

# ── CUDA check ────────────────────────────────────────────────
echo ""
log "Checking CUDA (optional — for GPU batch verification)..."
if command -v nvidia-smi &>/dev/null; then
    nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader
    if ! command -v nvcc &>/dev/null; then
        warn "nvcc not found. Install CUDA toolkit for GPU benchmark:"
        warn "  apt install nvidia-cuda-toolkit"
    fi
else
    warn "No NVIDIA GPU detected. GPU benchmark will use analytical model."
fi

# ── Record hardware info ──────────────────────────────────────
mkdir -p results/logs
python3 -c "
import platform, json
from datetime import datetime
try:
    import oqs
    liboqs_ver = oqs.oqs_version()
except:
    liboqs_ver = 'unknown'
info = {
    'generated':   datetime.now().isoformat(),
    'cpu':         platform.processor(),
    'os':          platform.platform(),
    'python':      platform.python_version(),
    'liboqs':      liboqs_ver,
    'role':        'gateway',
}
with open('results/logs/hardware_info.json','w') as f:
    json.dump(info, f, indent=2)
print('  Hardware info saved to results/logs/hardware_info.json')
" 2>/dev/null || true

echo ""
echo "======================================================"
log "Gateway setup complete."
echo ""
echo "  Activate environment:  source ~/pq_tdag_env/bin/activate"
echo "  Run first experiment:  python3 src/step_2_gateway_benchmark.py"
echo "======================================================"
