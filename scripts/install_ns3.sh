#!/bin/bash
# scripts/install_ns3.sh
#
# Installs NS-3.40 and applies the two patches required for PQ-TDAG.
#
# Patches applied automatically:
#   [1] Root-check bypass  — NS-3 refuses to run as root by default.
#       We locate the call site and replace it with a no-op.
#   [2] nr-module removal  — The CTTC 5G NR module requires eigen3 and
#       hosvd_deps which are non-standard. PQ-TDAG scenarios use only
#       PointToPointHelper and do not require the NR module.
#
# Usage:
#   bash scripts/install_ns3.sh
#
# After completion:
#   cd ~/ns-allinone-3.40/ns-3.40
#   ./ns3 build -j$(nproc)

set -e
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
log()  { echo -e "${GREEN}[ok]${NC} $1"; }
warn() { echo -e "${YELLOW}[!!]${NC} $1"; }

NS3_DIR="$HOME/ns-allinone-3.40/ns-3.40"

echo ""
echo "======================================================"
echo "  PQ-TDAG NS-3.40 Setup"
echo "======================================================"

# ── System dependencies ───────────────────────────────────────
log "Installing NS-3 build dependencies..."
apt-get update -qq
apt-get install -y \
    g++ cmake ninja-build git python3 \
    python3-dev pkg-config sqlite3 libsqlite3-dev \
    libxml2 libxml2-dev \
    tcpdump 2>/dev/null || true

# ── Download NS-3.40 ──────────────────────────────────────────
cd ~
if [ ! -d "ns-allinone-3.40" ]; then
    log "Downloading NS-3.40..."
    wget -q --show-progress \
        https://www.nsnam.org/releases/ns-allinone-3.40.tar.bz2
    tar xjf ns-allinone-3.40.tar.bz2
    log "Extracted."
else
    log "ns-allinone-3.40 already present."
fi

cd "$NS3_DIR"

# ── Patch [1]: disable root check ────────────────────────────
log "Applying patch [1]: root-check bypass..."
# Find the line number where refuse_run_as_root() is CALLED (not defined).
# The function definition appears earlier; we target the call inside main().
CALL_LINE=$(grep -n "refuse_run_as_root()" ns3 | tail -1 | cut -d: -f1)
if [ -n "$CALL_LINE" ]; then
    sed -i "${CALL_LINE}s/    refuse_run_as_root()/    pass  # patched by pq-tdag install/" ns3
    log "Root check disabled at line $CALL_LINE."
else
    warn "Could not find refuse_run_as_root() call. NS-3 may fail if running as root."
fi

# ── Patch [2]: remove nr-module ──────────────────────────────
log "Applying patch [2]: removing nr-module from contrib/..."
if [ -d "contrib/nr" ]; then
    mv contrib/nr ~/nr_backup
    log "nr-module moved to ~/nr_backup"
elif [ ! -d "contrib/nr" ]; then
    log "nr-module not present (already removed or not downloaded)."
fi

# Clean cmake cache after patch
rm -rf cmake-cache

# ── Configure NS-3 ────────────────────────────────────────────
log "Configuring NS-3 (optimized build)..."
./ns3 configure \
    --build-profile=optimized \
    --enable-examples \
    --disable-gtk \
    --disable-python

# ── Build NS-3 ────────────────────────────────────────────────
log "Building NS-3 — this takes 15-25 minutes on a modern CPU..."
./ns3 build -j$(nproc)

# ── Verify ────────────────────────────────────────────────────
log "Verifying build..."
HELLO="$NS3_DIR/build/examples/tutorial/ns3.40-hello-simulator-optimized"
if [ -f "$HELLO" ]; then
    $HELLO
    log "Hello simulator: OK (exit code $?)"
else
    warn "Could not find hello-simulator binary. Build may have failed."
fi

# ── Deploy PQ-TDAG scenarios ──────────────────────────────────
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
log "Deploying PQ-TDAG NS-3 scenarios..."
mkdir -p "$NS3_DIR/scratch/pq_tdag"
cp "$REPO_DIR/ns3/"*.cc "$NS3_DIR/scratch/pq_tdag/"
cp "$REPO_DIR/ns3/CMakeLists.txt" "$NS3_DIR/scratch/pq_tdag/"
cp "$REPO_DIR/ns3/run_all.sh"     "$NS3_DIR/scratch/pq_tdag/"
chmod +x "$NS3_DIR/scratch/pq_tdag/run_all.sh"

log "Building PQ-TDAG scenarios..."
./ns3 build -j$(nproc)

echo ""
echo "======================================================"
log "NS-3.40 setup complete."
echo ""
echo "  Run simulations:"
echo "    cd $NS3_DIR"
echo "    bash scratch/pq_tdag/run_all.sh"
echo ""
echo "  Copy results back:"
echo "    cp results/*.csv $REPO_DIR/results/data/"
echo "======================================================"
