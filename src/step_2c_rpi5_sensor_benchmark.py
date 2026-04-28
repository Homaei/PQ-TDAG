"""
src/step_2c_rpi5_sensor_benchmark.py
───────────────────────────────────────────────────────────────────
Sensor-side cryptographic benchmark for Raspberry Pi 5.

Architecture context:
  In the PQ-TDAG system, SENSORS sign data before transmitting it.
  GATEWAYS verify signatures. These two operations happen on entirely
  different hardware: signing on a constrained ARM edge node,
  verification on a server-class x86 gateway.

  Using the gateway CPU (i9-14900KF with AVX2) to model sensor
  latency is a methodological error — it would underestimate the
  signing time by roughly 32× and produce an optimistic energy
  estimate. This script corrects that by running directly on the RPi5.

Target hardware:
  Raspberry Pi 5
    CPU  : Broadcom BCM2712, Cortex-A76, quad-core, 2.4 GHz
    RAM  : 8 GB LPDDR4X
    ISA  : ARM Neon/ASIMD  (no AVX2, no AVX-512)
    OS   : Ubuntu 24.04 LTS  (aarch64)
  liboqs uses the ARM Neon code path for ML-DSA.

Instructions:
  1. Copy this file to the RPi5:
       scp src/step_2c_rpi5_sensor_benchmark.py hubert@<rpi5-ip>:~/
  2. On the RPi5:
       source ~/pq_tdag_env/bin/activate
       python3 step_2c_rpi5_sensor_benchmark.py
  3. Copy the output back:
       scp hubert@<rpi5-ip>:~/crypto_timings_rpi5.json results/data/

Output:
  ~/crypto_timings_rpi5.json  (on the RPi5)
  Copy to project: results/data/crypto_timings_rpi5.json
"""

import os, sys, json, time, gc, statistics, tracemalloc
from datetime import datetime
from pathlib import Path

import numpy as np

try:
    import oqs
except ImportError:
    sys.exit(
        "liboqs-python is not installed on this device.\n"
        "Run:  pip install liboqs-python"
    )

S_PAYLOAD  = 100    # ICS sensor payload (bytes)
N_ITER     = 200    # Fewer iterations than gateway: RPi5 is slower
WARMUP     = 10

# Schemes relevant for sensor-side signing.
# We omit naive_mldsa44 (same crypto as pq_tdag) and ecdsa (classical).
SENSOR_SCHEMES = [
    {"id": "mldsa44",    "oqs_name": "ML-DSA-44",              "label": "ML-DSA-44 (= PQ-TDAG crypto)"},
    {"id": "mldsa65",    "oqs_name": "ML-DSA-65",              "label": "ML-DSA-65 (FIPS204 L3)"},
    {"id": "falcon512",  "oqs_name": "Falcon-512",             "label": "Falcon-512"},
    {"id": "slhdsa128s", "oqs_name": "SLH_DSA_PURE_SHA2_128S", "label": "SLH-DSA-SHA2-128s (FIPS205)"},
    {"id": "slhdsa128f", "oqs_name": "SLH_DSA_PURE_SHA2_128F", "label": "SLH-DSA-SHA2-128f (FIPS205)"},
]


def detect_cpu_info() -> dict:
    """
    On ARM, the interesting ISA flags live under 'Features' in /proc/cpuinfo,
    not 'flags' as on x86. We look for neon/asimd (SIMD), sha2 (SHA accelerator),
    and aes (AES accelerator) — all of which liboqs can exploit.
    """
    info = {
        "model": "Raspberry Pi 5 (BCM2712, Cortex-A76)",
        "arch":  "aarch64",
        "cores": 4,
        "freq_ghz": 2.4,
        "instruction_sets": [],
        "platform": "rpi5",
    }
    try:
        import re
        with open("/proc/cpuinfo") as f:
            content = f.read()

        m = re.search(r"Model name\s*:\s*(.*)", content, re.IGNORECASE)
        if m:
            info["model"] = m.group(1).strip()

        # ARM features
        feat = re.search(r"Features\s*:\s*(.*)", content)
        if feat:
            all_flags = feat.group(1).split()
            wanted    = {"asimd", "neon", "aes", "sha1", "sha2", "sha3",
                         "crc32", "sve", "bf16", "i8mm", "dcpop"}
            info["instruction_sets"] = sorted([f for f in all_flags if f in wanted])

        # Read max frequency from sysfs
        freq_path = Path("/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq")
        if freq_path.exists():
            info["freq_ghz"] = round(int(freq_path.read_text().strip()) / 1e6, 2)

    except Exception as e:
        info["error"] = str(e)
    return info


def benchmark(oqs_name: str) -> dict:
    """
    Measures signing latency on the RPi5's Cortex-A76.

    We use 200 iterations (vs 500 on the gateway) because:
      - SLH-DSA-128s takes ~4 seconds per signing operation on the RPi5
      - 200 iterations × 4 s = ~13 minutes, which is the practical limit
      - ML-DSA-44 at 200 iterations takes ~0.3 seconds — acceptable

    The statistical estimates (mean, std, percentiles) remain reliable
    at 200 iterations because the variance of ML-DSA on ARM is low
    once the CPU is in steady state.
    """
    enabled  = oqs.get_enabled_sig_mechanisms()
    if oqs_name not in enabled:
        return {"error": f"not in liboqs {oqs.oqs_version()} on this device"}

    try:
        sig = oqs.Signature(oqs_name)
    except Exception as e:
        return {"error": str(e)}

    msg    = bytes(S_PAYLOAD)
    pk     = sig.generate_keypair()
    sk_len = len(sig.export_secret_key())

    # Warm-up
    for _ in range(WARMUP):
        s = sig.sign(msg)
        sig.verify(msg, s, pk)

    gc.collect()

    sign_t, verify_t = [], []

    for _ in range(N_ITER):
        t0 = time.perf_counter()
        signature = sig.sign(msg)
        sign_t.append((time.perf_counter() - t0) * 1000)

    for _ in range(N_ITER):
        t0 = time.perf_counter()
        sig.verify(msg, signature, pk)
        verify_t.append((time.perf_counter() - t0) * 1000)

    tracemalloc.start()
    tmp = oqs.Signature(oqs_name)
    pk2 = tmp.generate_keypair()
    s2  = tmp.sign(msg)
    tmp.verify(msg, s2, pk2)
    _, mem_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    tmp.free(); sig.free()

    return {
        "sig_bytes":          len(signature),
        "pk_bytes":           len(pk),
        "sk_bytes":           sk_len,
        "t_sign_mean_ms":     round(statistics.mean(sign_t),   4),
        "t_sign_std_ms":      round(statistics.stdev(sign_t),  4),
        "t_sign_p50_ms":      round(np.percentile(sign_t,  50),4),
        "t_sign_p95_ms":      round(np.percentile(sign_t,  95),4),
        "t_sign_p99_ms":      round(np.percentile(sign_t,  99),4),
        "t_verify_mean_ms":   round(statistics.mean(verify_t), 4),
        "t_verify_std_ms":    round(statistics.stdev(verify_t),4),
        "t_verify_p99_ms":    round(np.percentile(verify_t,99),4),
        "sign_ops_per_sec":   round(1000 / statistics.mean(sign_t),  1),
        "verify_ops_per_sec": round(1000 / statistics.mean(verify_t),1),
        "mem_peak_kb":        round(mem_peak / 1024, 2),
        "n_iterations":       N_ITER,
        "platform":           "Raspberry Pi 5 (Cortex-A76 @ 2.4 GHz)",
    }


def main():
    print("\n" + "=" * 60)
    print("  PQ-TDAG — RPi5 Sensor Benchmark")
    print("=" * 60 + "\n")

    cpu = detect_cpu_info()
    print(f"  Model : {cpu.get('model')}")
    print(f"  Freq  : {cpu.get('freq_ghz')} GHz")
    print(f"  ISA   : {', '.join(cpu.get('instruction_sets', []))}\n")

    results = {}
    for s in SENSOR_SCHEMES:
        sys.stdout.write(f"  {s['label']:<42}")
        sys.stdout.flush()
        r = benchmark(s["oqs_name"])
        r["id"]    = s["id"]
        r["label"] = s["label"]
        results[s["id"]] = r

        if "error" in r:
            print(f"  SKIP ({r['error']})")
        else:
            T_MAX    = 50.0
            ok_str   = "PASS" if r["t_sign_p99_ms"] < T_MAX else "FAIL (>> T_max)"
            print(
                f"  sign={r['t_sign_mean_ms']:.3f}ms "
                f"p99={r['t_sign_p99_ms']:.3f}ms  [{ok_str}]"
            )

    # Save to home directory (will be copied back to gateway)
    out = Path.home() / "crypto_timings_rpi5.json"
    with open(out, "w") as f:
        json.dump({
            "metadata": {
                "generated":  datetime.now().isoformat(),
                "platform":   "Raspberry Pi 5",
                "cpu":        cpu.get("model"),
                "freq_ghz":   cpu.get("freq_ghz"),
                "isa":        cpu.get("instruction_sets"),
                "liboqs":     oqs.oqs_version(),
                "n_iter":     N_ITER,
                "payload_bytes": S_PAYLOAD,
                "role":       "ICS_sensor_signing",
            },
            "timings": results,
        }, f, indent=2)

    print(f"\n  Saved: {out}")
    print()
    print("  Copy back to gateway:")
    print(f"  scp $(hostname -I | awk '{{print $1}}'):{out} results/data/")

    # Print summary
    print()
    print("  Summary vs T_max = 50 ms:")
    print(f"  {'Scheme':<18} {'t_sign (ms)':>12}  {'p99 (ms)':>10}  Result")
    print("  " + "-" * 55)
    for sid, r in results.items():
        if "error" in r: continue
        ok = "PASS" if r["t_sign_p99_ms"] < 50 else "FAIL"
        print(f"  {r['label']:<18}  {r['t_sign_mean_ms']:>10.3f}  {r['t_sign_p99_ms']:>10.3f}  {ok}")


if __name__ == "__main__":
    main()
