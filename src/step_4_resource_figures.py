"""
step_4_group_E_resources.py
══════════════════════════════════════════════════════════════════
Group E: Resource Consumption Figures
  fig_E1 — Memory Footprint per Gateway vs N
  fig_E2 — Energy per Transaction (all 8 schemes)
  fig_E3 — DAG Storage Growth over Time

Reads:  results/data/crypto_timings.json
Output: results/figures/fig_E*.pdf

Run:
  python3 src/step_4_group_E_resources.py
══════════════════════════════════════════════════════════════════
"""

import json, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

ROOT        = Path(__file__).parent.parent
DATA_FILE   = ROOT / "results/data/crypto_timings.json"
FIGURES_DIR = ROOT / "results/figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family":    "serif",
    "font.size":      11,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "legend.fontsize":9,
    "xtick.labelsize":10,
    "ytick.labelsize":10,
    "figure.dpi":     150,
    "axes.grid":      True,
    "grid.alpha":     0.35,
    "grid.linestyle": "--",
    "lines.linewidth":1.8,
    "lines.markersize":6,
})

# ── Physical parameters ───────────────────────────────────────
S_PAYLOAD_B     = 100
F_HZ_REF        = 20
M_WINDOW_REF    = 5
N_RANGE         = np.array([10, 25, 50, 100, 200, 300, 500, 750, 1000])
SIM_MINUTES     = 60

# ── Energy model (µJ) — from ARM Cortex-M4 + i7 benchmarks ──
# Source: pqm4 benchmark suite + NIST PQC energy measurements
ENERGY_MODEL = {
    # (sign_uJ, hash_per_byte_uJ, verify_uJ, transmit_per_byte_uJ)
    "pq_tdag":     (8.4,  0.0003, 0.315, 0.0008),
    "naive_mldsa44":(8.4, 0.0003, 0.315, 0.0008),
    "mldsa65":     (14.7, 0.0003, 0.490, 0.0008),
    "falcon512":   (26.6, 0.0003, 0.070, 0.0008),
    "slhdsa128s":  (406., 0.0003, 12.60, 0.0008),
    "slhdsa128f":  (22.4, 0.0003, 24.50, 0.0008),
    "xmssmt":      (101.5,0.0003, 8.40,  0.0008),
    "ecdsa":       (0.35, 0.0003, 0.035, 0.0008),
}


def load_timings():
    if not DATA_FILE.exists():
        print(f"ERROR: {DATA_FILE} not found. Run Step 2 first.")
        sys.exit(1)
    with open(DATA_FILE) as f:
        data = json.load(f)
    return data["schemes"]


# ══════════════════════════════════════════════════════════════
#  MEMORY MODEL
# Each gateway must store:
#   - DAG index:        N * avg_dag_entries * entry_size
#   - Key table:        N * (pk_bytes + id_bytes)
#   - Verification buf: M * (S_payload + sig_bytes)  per active chain
#   - Out-of-order buf: TBFR buffer
# ══════════════════════════════════════════════════════════════
def gateway_memory_mb(n_sensors, sig_bytes, pk_bytes, M=5,
                      dag_depth=200):
    """
    Estimate RAM usage per edge gateway (MB).
    dag_depth: average DAG entries kept in active window.
    """
    # DAG active window: N sensors × dag_depth entries × entry_size
    entry_size    = S_PAYLOAD_B + 32 + 8   # payload + hash + metadata
    dag_memory    = n_sensors * dag_depth * entry_size

    # Public key table
    key_table     = n_sensors * (pk_bytes + 64)   # pk + sensor ID + cert

    # Verification buffer (one per active micro-chain)
    verify_buf    = n_sensors * M * (S_PAYLOAD_B + 32)

    # TBFR out-of-order buffer (worst case: M packets per sensor)
    tbfr_buf      = n_sensors * M * S_PAYLOAD_B

    # Signature cache (recently verified)
    sig_cache     = n_sensors * 10 * sig_bytes

    total_bytes   = dag_memory + key_table + verify_buf + tbfr_buf + sig_cache
    return total_bytes / (1024 * 1024)   # to MB


# ══════════════════════════════════════════════════════════════
#  FIGURE E1 — Memory vs N
# ══════════════════════════════════════════════════════════════
def plot_E1_memory_vs_n(schemes: dict):
    print("  Plotting fig_E1: Memory vs N...")

    fig, ax = plt.subplots(figsize=(7, 5))

    plot_ids = ["pq_tdag", "naive_mldsa44", "falcon512",
                "mldsa65", "slhdsa128s", "xmssmt", "ecdsa"]

    for sid in plot_ids:
        if sid not in schemes:
            continue
        sdata  = schemes[sid]
        sig_b  = sdata.get("sig_bytes", 0)
        pk_b   = sdata.get("pk_bytes", 64)
        label  = sdata.get("label", sid)
        color  = sdata.get("color", "gray")
        marker = sdata.get("marker", "o")
        M      = M_WINDOW_REF if sid == "pq_tdag" else 1

        mem = [gateway_memory_mb(n, sig_b, pk_b, M) for n in N_RANGE]

        lw = 2.5 if sid == "pq_tdag" else 1.5
        ax.plot(N_RANGE, mem, color=color, marker=marker,
                linewidth=lw, markersize=6 if sid == "pq_tdag" else 5,
                label=label)

    # Typical edge gateway RAM limits
    for limit, lbl in [(256, "256 MB"), (512, "512 MB"), (1024, "1 GB")]:
        ax.axhline(limit, linestyle=":", linewidth=1.0,
                   color="gray", alpha=0.6)
        ax.text(N_RANGE[-1] * 0.98, limit + 5, lbl,
                ha="right", va="bottom", fontsize=8, color="gray")

    ax.set_xlabel("Number of Sensors $N$")
    ax.set_ylabel("Gateway RAM Footprint (MB)")
    ax.set_title("(E1) Memory Footprint per Edge Gateway vs. $N$")
    ax.set_xlim(N_RANGE[0], N_RANGE[-1])
    ax.set_ylim(0, 1200)
    ax.legend(fontsize=8.5, ncol=1, framealpha=0.9)

    out = FIGURES_DIR / "fig_E1_memory_vs_n.pdf"
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight", dpi=300)
    plt.savefig(str(out).replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {out}")


# ══════════════════════════════════════════════════════════════
#  FIGURE E2 — Energy per Transaction
# ══════════════════════════════════════════════════════════════
def energy_per_tx_uj(sid, sig_bytes, M=1):
    """
    Total energy per verified ICS transaction (µJ).
    For PQ-TDAG: sign cost amortized over M transactions.
    Components: sign (amortized) + hash chain + transmit + verify
    """
    e_sign_uj, e_hash_uj_per_byte, e_verify_uj, e_tx_uj_per_byte = \
        ENERGY_MODEL.get(sid, (10.0, 0.0003, 1.0, 0.0008))

    e_sign_amort   = e_sign_uj / M
    e_hash_chain   = e_hash_uj_per_byte * S_PAYLOAD_B * M / M  # per tx
    e_transmit     = e_tx_uj_per_byte * (S_PAYLOAD_B + sig_bytes / M)
    e_verify_amort = e_verify_uj / M

    return e_sign_amort + e_hash_chain + e_transmit + e_verify_amort


def plot_E2_energy_per_tx(schemes: dict):
    print("  Plotting fig_E2: Energy per Transaction...")

    fig, ax = plt.subplots(figsize=(9, 5))

    order = ["ecdsa", "pq_tdag", "naive_mldsa44", "falcon512",
             "mldsa65", "slhdsa128f", "xmssmt", "slhdsa128s"]

    labels_plot, energies, colors_plot = [], [], []

    for sid in order:
        if sid not in schemes:
            continue
        sdata   = schemes[sid]
        sig_b   = sdata.get("sig_bytes", 64)
        M       = M_WINDOW_REF if sid == "pq_tdag" else 1
        e       = energy_per_tx_uj(sid, sig_b, M)
        label   = sdata.get("label", sid)
        color   = sdata.get("color", "gray")
        labels_plot.append(label)
        energies.append(e)
        colors_plot.append(color)

    x   = np.arange(len(labels_plot))
    bar = ax.bar(x, energies, 0.65, color=colors_plot,
                 edgecolor="black", linewidth=0.6)

    # Highlight our method
    our_idx = next((i for i, s in enumerate(order) if s == "pq_tdag"), None)
    if our_idx is not None:
        bar[our_idx].set_edgecolor("red")
        bar[our_idx].set_linewidth(2.5)

    # Value labels on top
    for rect, val in zip(bar, energies):
        ax.text(rect.get_x() + rect.get_width() / 2,
                rect.get_height() + 0.5,
                f"{val:.1f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels_plot, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("Energy per Transaction ($\\mu$J)")
    ax.set_title(
        "(E2) Energy per Verified Transaction\n"
        f"$N={50}$ sensors, $f={F_HZ_REF}$ Hz, "
        f"$M={M_WINDOW_REF}$ (PQ-TDAG), $M=1$ (others)"
    )
    ax.set_ylim(0, max(energies) * 1.2)

    out = FIGURES_DIR / "fig_E2_energy_per_tx.pdf"
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight", dpi=300)
    plt.savefig(str(out).replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {out}")


# ══════════════════════════════════════════════════════════════
#  FIGURE E3 — DAG Storage Growth
# ══════════════════════════════════════════════════════════════
def dag_storage_mb(t_minutes, N, f, sig_bytes, pk_bytes, M,
                   is_pqtdag=False):
    """
    Cumulative DAG storage in MB over t_minutes.
    """
    t_seconds      = t_minutes * 60
    total_payloads = N * f * t_seconds * S_PAYLOAD_B

    if is_pqtdag:
        # Terminal nodes only signed; internal nodes: hash only (32B)
        n_terminal   = N * (f / M) * t_seconds
        n_internal   = N * f * t_seconds * (M - 1) / M
        storage_sigs = n_terminal * sig_bytes
        storage_data = n_internal * (S_PAYLOAD_B + 32)   # payload + hash
        storage_term = n_terminal * (S_PAYLOAD_B + sig_bytes + 64)  # + DAG meta
        total        = storage_data + storage_term
    else:
        # Naive: every transaction fully signed
        total = N * f * t_seconds * (S_PAYLOAD_B + sig_bytes + 64)

    return total / (1024 * 1024)


def plot_E3_dag_storage(schemes: dict):
    print("  Plotting fig_E3: DAG Storage Growth...")

    times = np.linspace(0, SIM_MINUTES, 200)

    fig, ax = plt.subplots(figsize=(7, 5))

    plot_ids = ["pq_tdag", "naive_mldsa44", "falcon512",
                "mldsa65", "xmssmt", "ecdsa"]

    for sid in plot_ids:
        if sid not in schemes:
            continue
        sdata  = schemes[sid]
        sig_b  = sdata.get("sig_bytes", 0)
        pk_b   = sdata.get("pk_bytes", 64)
        label  = sdata.get("label", sid)
        color  = sdata.get("color", "gray")
        is_our = (sid == "pq_tdag")
        M      = M_WINDOW_REF if is_our else 1

        storage = [dag_storage_mb(t, N=50, f=F_HZ_REF,
                                  sig_bytes=sig_b, pk_bytes=pk_b,
                                  M=M, is_pqtdag=is_our)
                   for t in times]

        lw = 2.5 if is_our else 1.5
        ax.plot(times, storage, color=color, linewidth=lw, label=label)

    ax.set_xlabel("Simulation Time (minutes)")
    ax.set_ylabel("Cumulative DAG Storage (MB)")
    ax.set_title(
        f"(E3) DAG Storage Growth over Time\n"
        f"$N=50$ sensors, $f={F_HZ_REF}$ Hz, "
        f"$M={M_WINDOW_REF}$ (PQ-TDAG), $M=1$ (others)"
    )
    ax.set_xlim(0, SIM_MINUTES)
    ax.legend(fontsize=8.5, ncol=1, framealpha=0.9)

    out = FIGURES_DIR / "fig_E3_dag_storage.pdf"
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight", dpi=300)
    plt.savefig(str(out).replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {out}")


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════
def main():
    print()
    print("═" * 60)
    print("  PQ-TDAG — Group E: Resource Figures (Step 4)")
    print("═" * 60)
    print()

    schemes = load_timings()
    print(f"  Loaded {len(schemes)} schemes")
    print()

    plot_E1_memory_vs_n(schemes)
    plot_E2_energy_per_tx(schemes)
    plot_E3_dag_storage(schemes)

    print()
    print("═" * 60)
    print("  DONE — Group E figures saved.")
    print()
    print("  Next step:")
    print("  python3 src/step_5_group_D_security.py")
    print("═" * 60)
    print()


if __name__ == "__main__":
    main()
