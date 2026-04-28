#!/bin/bash
# run_all_ns3.sh
# Run all PQ-TDAG NS-3 simulations.
# Must be executed from NS-3 root:
#   cd ~/ns-allinone-3.40/ns-3.40
#   bash scratch/pq_tdag/run_all_ns3.sh

set -e
mkdir -p results

echo "══════════════════════════════════════════════"
echo "  PQ-TDAG NS-3 Simulation Suite"
echo "══════════════════════════════════════════════"

# Latency CDF — 6 schemes (~29k data points each)
echo "[C1] Latency CDF..."
for s in pq_tdag naive_mldsa44 falcon512 mldsa65 slhdsa128f ecdsa; do
    echo "  scheme=$s"
    ./ns3 run "pqtdag_pqtdag_latency_cdf --scheme=$s"
done

# Throughput vs M and vs N
echo "[B1,B2] Throughput..."
for s in pq_tdag naive_mldsa44 falcon512 mldsa65; do
    echo "  window sweep: $s"
    ./ns3 run "pqtdag_pqtdag_throughput --scheme=$s --mode=window"
    echo "  scale sweep: $s"
    ./ns3 run "pqtdag_pqtdag_throughput --scheme=$s --mode=scale"
done

# Erasure resilience (TBFR)
echo "[C2] Erasure..."
for M in 3 5 8 10; do
    echo "  M=$M"
    ./ns3 run "pqtdag_pqtdag_erasure --M=$M"
done

echo ""
echo "══════════════════════════════════════════════"
echo "  All simulations complete."
echo "  Results in: results/"
echo "  Copy to project: cp results/*.csv /path/to/pq_tdag/results/data/"
echo "══════════════════════════════════════════════"
