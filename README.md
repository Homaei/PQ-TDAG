# PQ-TDAG: Bandwidth-Efficient Post-Quantum Authentication for Industrial Control Systems via Micro-Chaining

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://python.org)
[![NS-3 3.40](https://img.shields.io/badge/NS--3-3.40-green.svg)](https://nsnam.org)
[![liboqs 0.15.0](https://img.shields.io/badge/liboqs-0.15.0-orange.svg)](https://github.com/open-quantum-safe/liboqs)

> **Submitted to:** IEEE (Pre-Print)

---

## Overview

Industrial Control Systems (ICS) face an approaching quantum threat: their classical cryptographic schemes (ECDSA, RSA) will be broken by sufficiently powerful quantum computers. The post-quantum replacement candidates standardised by NIST — ML-DSA-44 (FIPS 204), SLH-DSA (FIPS 205) — have signatures 20–300× larger than ECDSA-P256, making naive deployment infeasible on 5G URLLC links within ICS latency budgets.

**PQ-TDAG** resolves this conflict through a micro-chaining architecture: sensors sign one *terminal node* for every `M` *internal nodes*, amortising the signature overhead across `M` transactions. The result is a post-quantum authenticated DAG that fits within the 10 Mbps 5G URLLC channel budget while meeting the T_max = 50 ms ICS control-loop deadline.

### Core Empirical Results

| Metric | PQ-TDAG (M=5) | Naive ML-DSA-44 | Improvement |
|--------|--------------|-----------------|-------------|
| Required bandwidth | **4.67 Mbps** | 20.16 Mbps | **4.3× less** |
| Max feasible sensors | **≥ 1000** | 25 | **40× more** |
| Energy per transaction | **1.82 µJ** | 8.79 µJ | **4.8× less** |
| TBFR delivery at p_e=10% | **100%** | — | deadline met |

Both schemes use identical ML-DSA-44 cryptography. Every gain is architectural.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  ICS FIELD LAYER                                            │
│  Sensor: Raspberry Pi 5  /  NVIDIA Jetson Orin Nano         │
│  Operation: SIGN (ML-DSA-44) → t_sign on ARM hardware       │
└─────────────────┬───────────────────────────────────────────┘
                  │  5G URLLC  (B_max = 10 Mbps, T_max = 50 ms)
┌─────────────────▼───────────────────────────────────────────┐
│  EDGE GATEWAY LAYER                                         │
│  Platform: Intel Core i9-14900KF  /  NVIDIA RTX 4070        │
│  Operation: VERIFY (batch) + DAG tip selection + TBFR       │
└─────────────────────────────────────────────────────────────┘
```

---

## Repository Structure

```
pq-tdag/
├── src/
│   ├── step_2_gateway_benchmark.py      # Gateway crypto timings (liboqs)
│   ├── step_2b_fix_liboqs_015.py        # liboqs 0.15.0 name-change fix
│   ├── step_2c_rpi5_sensor_benchmark.py # Run on physical RPi5
│   ├── step_2d_jetson_sensor_benchmark.py # Run on physical Jetson
│   ├── step_3_bandwidth_figures.py      # Group A: bandwidth analysis
│   ├── step_4_resource_figures.py       # Group E: memory, energy, storage
│   ├── step_5_security_figures.py       # Group D: Byzantine, TBFR, attack
│   ├── step_6_ns3_plot_results.py       # Group B/C: plot NS-3 CSV output
│   ├── step_7_gpu_benchmark.py          # GPU batch verification
│   ├── step_8_final_tables.py           # LaTeX table generator
│   └── gpu_batch_verify.cu              # CUDA source (RTX 4070 / Jetson)
├── ns3/
│   ├── pqtdag_latency_cdf.cc            # Latency CDF scenario
│   ├── pqtdag_throughput.cc             # Throughput vs M and N
│   ├── pqtdag_erasure.cc                # TBFR erasure resilience
│   ├── CMakeLists.txt                   # NS-3 build file
│   └── run_all.sh                       # Batch runner
├── scripts/
│   ├── install_gateway.sh               # Setup for i9-14900KF gateway
│   ├── install_sensor.sh                # Setup for RPi5 / Jetson
│   └── install_ns3.sh                   # NS-3.40 installation
├── plot_figures.py                      # Regenerate all 17 figures
├── requirements.txt
├── LICENSE
└── results/                             # Populated after running experiments
    ├── data/
    ├── figures/
    ├── tables/
    └── logs/
```

---

## Hardware Requirements

### Gateway (verification + DAG)

| Component | Minimum | Used in paper |
|-----------|---------|---------------|
| CPU | Any x86-64, 4+ cores | Intel Core i9-14900KF |
| RAM | 16 GB | 64 GB |
| GPU (optional) | Any CUDA 8.x | NVIDIA RTX 4070 |
| OS | Ubuntu 22.04+ | Ubuntu 24.04 LTS |

> **Note on ISA:** The i9-14900KF (Raptor Lake Refresh) uses AVX2 + AVX-VNNI.
> It does **not** have AVX-512 — this is a consumer SKU with AVX-512 fused off.
> All liboqs timings in the paper use the AVX2 code path.

### Sensor Platforms

| Platform | CPU | RAM | Role |
|----------|-----|-----|------|
| Raspberry Pi 5 | Cortex-A76, 2.4 GHz, 4-core | 8 GB | ICS sensor (primary) |
| NVIDIA Jetson Orin Nano | Cortex-A78AE, 1.5 GHz, 6-core + GA10B GPU | 8 GB | ICS sensor / edge GW |

---

## Installation

### Option A — Gateway (i9-14900KF or equivalent x86)

```bash
git clone https://github.com/your-org/pq-tdag.git
cd pq-tdag
chmod +x scripts/install_gateway.sh
bash scripts/install_gateway.sh
source ~/pq_tdag_env/bin/activate
```

### Option B — Sensor (Raspberry Pi 5)

```bash
git clone https://github.com/your-org/pq-tdag.git
cd pq-tdag
chmod +x scripts/install_sensor.sh
bash scripts/install_sensor.sh
source ~/pq_tdag_env/bin/activate
```

### Option C — Sensor (Jetson Orin Nano)

Same as Option B. JetPack 6.x includes CUDA; the GPU benchmark compiles automatically.

---

## Running the Experiments

All steps are independent; results from earlier steps feed into later ones.

### Step 1 — Gateway Crypto Benchmarks

Measures real signing and verification latency for all 8 schemes on the gateway CPU.

```bash
cd /home/hubert/pq-tdag
source ~/pq_tdag_env/bin/activate

python3 src/step_2_gateway_benchmark.py
# Runtime: 3-8 minutes
# Output: results/data/crypto_timings.json
#         results/logs/benchmark_report.txt   ← paste into paper Section 5.1
```

### Step 2 — Fix liboqs 0.15.0 Naming

liboqs 0.15.0 renamed scheme identifiers and dropped XMSS-MT from its build.
This step patches the JSON and injects RFC 8391 reference values for XMSS-MT.

```bash
python3 src/step_2b_fix_liboqs_015.py
# Runtime: 3-5 minutes (SLH-DSA-128s is slow to benchmark)
# Output: results/data/crypto_timings.json  (updated, 8 schemes complete)
```

### Step 3 — Sensor Benchmarks (run on physical hardware)

**On Raspberry Pi 5:**
```bash
scp src/step_2c_rpi5_sensor_benchmark.py hubert@<rpi5-ip>:~/
ssh hubert@<rpi5-ip>
source ~/pq_tdag_env/bin/activate
python3 step_2c_rpi5_sensor_benchmark.py
# Output: crypto_timings_rpi5.json  — copy back to results/data/
```

**On Jetson Orin Nano:**
```bash
scp src/step_2d_jetson_sensor_benchmark.py hubert@<jetson-ip>:~/
ssh hubert@<jetson-ip>
source ~/pq_tdag_env/bin/activate
python3 step_2d_jetson_sensor_benchmark.py
# Output: crypto_timings_jetson.json  — copy back to results/data/
```

### Step 4 — Analytical Figures (no NS-3 required)

```bash
python3 src/step_3_bandwidth_figures.py   # fig_A1, A2, A3
python3 src/step_4_resource_figures.py    # fig_E1, E2, E3
python3 src/step_5_security_figures.py    # fig_D1, D2, D3  (~10 min Monte Carlo)
```

### Step 5 — NS-3 Network Simulation

Install NS-3 first (one-time, ~25 minutes):

```bash
bash scripts/install_ns3.sh
```

Deploy and run:

```bash
mkdir -p /root/ns-allinone-3.40/ns-3.40/scratch/pq_tdag
cp ns3/*.cc ns3/CMakeLists.txt /root/ns-allinone-3.40/ns-3.40/scratch/pq_tdag/
cd /root/ns-allinone-3.40/ns-3.40
./ns3 build -j$(nproc)
bash scratch/pq_tdag/run_all.sh
cp results/*.csv /home/hubert/pq-tdag/results/data/
cd /home/hubert/pq-tdag
python3 src/step_6_ns3_plot_results.py    # fig_B1, B2, C1, C2, C3
```

### Step 6 — GPU Benchmark (optional, RTX 4070 / Jetson)

```bash
python3 src/step_7_gpu_benchmark.py
# If CUDA available: compiles and runs gpu_batch_verify.cu automatically
# Otherwise: uses validated RTX 4070 analytical model
```

### Step 7 — Final Tables

```bash
python3 src/step_8_final_tables.py
# Output: results/tables/table1_baselines.tex
#         results/tables/table2_security.tex
```

### Regenerate All Figures

After all data is collected:

```bash
python3 plot_figures.py
# All 17 figures written to results/figures/
# Edit the STYLE dict at the top to change fonts, colors, line widths
```

---

## Baselines

Eight schemes are evaluated. The selection is deliberate.

| ID | Scheme | Standard | σ (B) | Scientific role |
|----|--------|----------|-------|-----------------|
| pq_tdag | PQ-TDAG M=5 | ML-DSA-44 | 2420 | Proposed method |
| naive_mldsa44 | Naive ML-DSA-44 | FIPS 204 | 2420 | Isolates architecture contribution |
| mldsa65 | ML-DSA-65 | FIPS 204 L3 | 3309 | Higher-security trade-off |
| falcon512 | Falcon-512 | NIST Rd. 3 | 655 | Smallest PQC signature |
| slhdsa128s | SLH-DSA-128s | FIPS 205 slow | 7856 | Stateless — catastrophically slow |
| slhdsa128f | SLH-DSA-128f | FIPS 205 fast | 17088 | Stateless — bandwidth infeasible |
| xmssmt | XMSS-MT | RFC 8391 | 4963 | Stateful — formally proven unsuitable |
| ecdsa | ECDSA-P256 | FIPS 186-5 | 64 | Classical upper-bound reference |

> **Why XMSS-MT?** It is included to provide formal evidence that *stateful* signatures
> are operationally impossible in a distributed DAG with n ≥ 2 gateways:
> state synchronisation at 1000 tx/s requires 2000 ms/s of round-trip overhead,
> violating T_max = 50 ms by a factor of 40. See `docs/formal_proofs.md`.

---

## Key Results

### Measured (i9-14900KF, liboqs 0.15.0, 500 iterations)

| Scheme | σ (B) | t_sign (ms) | t_verify (ms) | B_req (Mbps) |
|--------|-------|-------------|---------------|--------------|
| **PQ-TDAG M=5** | 2420 | 0.044 ±0.027 | 0.016 ±0.001 | **4.67** |
| Naive ML-DSA-44 | 2420 | 0.043 ±0.025 | 0.016 ±0.001 | 20.16 |
| ML-DSA-65 | 3309 | 0.070 ±0.039 | 0.025 ±0.001 | 27.27 |
| Falcon-512 | 655 | 0.117 ±0.003 | 0.022 ±0.001 | 6.04 |
| SLH-DSA-128s | 7856 | 284.21 ±2.11 | 0.286 ±0.006 | 63.65 |
| SLH-DSA-128f | 17088 | 13.57 ±0.20 | 0.795 ±0.022 | 137.50 |
| XMSS-MT [†] | 4963 | 12.40 ±0.85 | 1.180 ±0.042 | 40.50 |
| ECDSA-P256 | 64 | 0.303 ±0.006 | 0.677 ±0.012 | 1.31 |

[†] Literature reference (RFC 8391 + pqm4). Not in liboqs 0.15.0.

### NS-3 Scalability

| Scheme | Max feasible N | Collapse point |
|--------|---------------|----------------|
| **PQ-TDAG** | **≥ 1000** | Never in test range |
| Naive ML-DSA-44 | 25 | N=25 |
| Falcon-512 | 100 | N=100 |

---

## Troubleshooting

> [!NOTE]
> **liboqs warnings on startup**
> 
> `UserWarning: liboqs version (major, minor) 0.15.0 differs from liboqs-python version 0.14.0`
> 
> **Cause:** The Python wrapper (0.14.0) and the underlying C library (0.15.0) have different version numbers.
> **Fix:** None. All measurements are valid. This is a cosmetic warning.

> [!WARNING]
> **SLH-DSA / SPHINCS+ scheme not found**
> 
> `ERROR: Not in this liboqs build. Available: check results/logs/`
> 
> **Cause:** liboqs 0.15.0 renamed scheme identifiers.
> - `SPHINCS+-SHA2-128s-simple` $\to$ `SLH_DSA_PURE_SHA2_128S`
> - `SPHINCS+-SHA2-128f-simple` $\to$ `SLH_DSA_PURE_SHA2_128F`
> 
> **Fix:** Run `step_2b_fix_liboqs_015.py` — it handles this automatically.

> [!NOTE]
> **XMSS-MT not found in liboqs**
> 
> **Cause:** XMSS-MT was removed from the liboqs 0.15.0 production build.
> **Fix:** `step_2b_fix_liboqs_015.py` injects reference values from RFC 8391 and marks them `[†]` in all tables. No manual action needed.

> [!CAUTION]
> **NS-3 refuses to run as root**
> 
> `Exception: Refusing to run as root. --enable-sudo will request your password when needed`
> 
> **Cause:** The NS-3 launcher script has a hard-coded root check.
> **Fix:** Run `scripts/install_ns3.sh` — it patches this automatically. If fixing manually, replace `refuse_run_as_root()` with `pass` in `~/ns-allinone-3.40/ns-3.40/ns3`.

> [!WARNING]
> **NS-3 configure fails: nr-module CMake error**
> 
> `CMake Error: Unknown CMake command "disable_cmake_warnings"`
> 
> **Cause:** The CTTC 5G NR module requires `eigen3` and `hosvd_deps`. Our scenarios use `PointToPointHelper` only.
> **Fix:** Run `scripts/install_ns3.sh` — it moves the NR module out of `contrib/` before configuring.

> [!NOTE]
> **NS-3 optimized build shows no output**
> 
> `ninja: no work to do.`
> 
> **Cause:** The `--build-profile=optimized` flag disables all `NS_LOG` output. The binary ran successfully and wrote output to `results/*.csv`.
> **Verify:** `ls /root/ns-allinone-3.40/ns-3.40/results/*.csv`

> [!WARNING]
> **NS-3 compile errors**
> 
> `error: reference to 'TimestampTag' is ambiguous` OR `error: cannot convert 'Callback<void, Ptr<const Packet>, const Address&>'`
> 
> **Cause:** NS-3.40 added `ns3::TimestampTag` and changed `Socket::SetRecvCallback` signature.
> **Fix:** The C++ files in `ns3/` already contain fixes (`PqTdagTag` and new callbacks). If using old files, run `python3 fix_ns3_compile.py`.

> [!NOTE]
> **CPU frequency reads 0.8 GHz**
> 
> `Freq   : 0.8 GHz`
> 
> **Cause:** `/proc/cpuinfo` reports idle frequency, not boost.
> **Fix:** `step_2b_fix_liboqs_015.py` reads the boost frequency from sysfs. Correct value for i9-14900KF is 4.4 GHz.

> [!NOTE]
> **SLH-DSA-128s t_sign >> T_max**
> 
> `SLH-DSA-SHA2-128s: t_sign = 284.213 ms >> T_max = 50 ms`
> 
> **Cause/Fix:** This is correct and expected. It demonstrates that stateless hash-based signatures are infeasible in real-time ICS.

---

## Reproducibility — Google Colab

All figures in the paper can be reproduced without installing liboqs or NS-3 by running the provided Colab notebook against 
the pre-computed experimental data.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/your-org/pq-tdag/blob/main/PQ_TDAG_Colab.ipynb)

**Steps:**
1. Open `PQ_TDAG_Colab.ipynb` in Google Colab
2. Upload `PQ-TDAG-main.zip` when prompted (Cell 1)
3. Run all cells — figures save to `results/figures/`
4. Last cell downloads all PDFs as a single zip

**Figures generated:** A1, A2, A3, A4, B1, B2, C1, C2, 
D1, D4, E2, sensitivity_tpipe  
**Runtime:** ~30 seconds  
**No GPU required**

---
## Citation

If you use this code or results, please cite:

```bibtex
@article{pqtdag2025,
  title   = {{PQ-TDAG}: Bandwidth-Efficient Post-Quantum Authentication
             for Industrial Control Systems via Micro-Chaining},
  authors = {Homaei. Mohammadhossein},
  journal = {IEEE Transactions on Industrial Informatics},
  year    = {2026}
}
```

---

## License

MIT License — see [LICENSE](LICENSE).

---

## Acknowledgements

- [Open Quantum Safe / liboqs](https://github.com/open-quantum-safe/liboqs) — PQC implementation library
- [NS-3 Network Simulator](https://nsnam.org) — network layer evaluation
- [pqm4](https://github.com/mupq/pqm4) — ARM energy model reference values
- [pqax](https://github.com/mupq/pqax) — ARM Neon NTT performance reference
