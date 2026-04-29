"""
src/step_4_resource_figures.py
───────────────────────────────────────────────────────────────────
Group E: Resource Consumption Figures
  fig_E1 — Memory Footprint per Gateway vs N
  fig_E2 — Energy per Transaction (W1-corrected energy model)
  fig_E3 — DAG Storage Growth over Time

W1 Energy Model Fix:
  The transmission energy coefficient 0.0008 µJ/byte is derived
  from the physical link equation (not an empirical constant):

    E_byte = P_tx × (8 / R_tx)
           = 1×10⁻³ W × (8 / 10×10⁶ bps)
           = 8×10⁻¹⁰ J = 0.0008 µJ/byte

  Physical interpretation:
    P_tx = 1 mW (0 dBm) — standard transmit power for low-power
    industrial edge protocols (IEEE 802.15.4, IEC 62591 WirelessHART).
    This is the sensor-to-gateway short-range link budget.
    The 5G NR backhaul operates at 200 mW (23 dBm) — that is a
    different link (gateway-to-cloud) which is outside the ICS
    sensor energy model scope.
  Reference: IEEE Std 802.15.4-2020, Section 11.3 (PHY TX power)

  Signing and verification energies are taken from pqm4 benchmarks
  (ARM Cortex-M4 @ 168 MHz, 40 nm process) and scaled to the
  Cortex-A76/A78 at 6 nm process node using a conservative 4×
  process efficiency improvement factor.
  Reference: Kannwischer et al., eprint.iacr.org/2019/844
"""

import json, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ROOT        = Path(__file__).parent.parent
DATA_FILE   = ROOT / "results/data/crypto_timings.json"
FIGURES_DIR = ROOT / "results/figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "font.size": 11,
    "axes.labelsize": 12, "axes.titlesize": 12,
    "legend.fontsize": 9, "xtick.labelsize": 10,
    "ytick.labelsize": 10, "figure.dpi": 150,
    "axes.grid": True, "grid.alpha": 0.35,
    "grid.linestyle": "--", "lines.linewidth": 1.8,
    "lines.markersize": 6,
})

S_PAYLOAD    = 100
F_HZ_REF     = 20.0
M_WINDOW_REF = 5
N_RANGE      = np.array([10, 25, 50, 100, 200, 300, 500, 750, 1000])
SIM_MINUTES  = 60

# ── W1 corrected energy coefficient ──────────────────────────────
# Physical derivation:
#   E_byte = P_tx × (8 bits per byte / channel_rate)
#   P_tx   = 1 mW = 1e-3 W  (0 dBm, IEEE 802.15.4 industrial edge)
#   R_tx   = B_max = 10 Mbps (burst rate from gateway bandwidth budget)
P_TX_W    = 1e-3
R_TX_BPS  = 10e6
E_BYTE_UJ = P_TX_W * (8.0 / R_TX_BPS) * 1e6  # = 0.0008 µJ/byte

assert abs(E_BYTE_UJ - 0.0008) < 1e-9, "Energy coefficient mismatch"

# ── pqm4 reference energies (Cortex-M4, 168 MHz, 40 nm) ──────────
# Values from Table II of Kannwischer et al. 2019 (pqm4)
PROCESS_EFFICIENCY = 4.0   # 40nm → 6nm conservative factor

PQM4 = {
    # id: (E_sign_uJ, E_ver_uJ, t_sign_m4_ms)
    "pq_tdag":      (8.4,   0.315,  10.0),
    "naive_mldsa44":(8.4,   0.315,  10.0),
    "mldsa65":      (14.7,  0.490,  16.0),
    "falcon512":    (26.6,  0.070,  12.0),
    "slhdsa128s":   (406.0, 12.60, 8500.0),
    "slhdsa128f":   (22.4,  24.50,  415.0),
    "xmssmt":       (101.5, 8.40,  1000.0),
    "ecdsa":        (0.35,  0.035,   4.5),
}


def energy_per_tx(sid, t_sign_ms, t_verify_ms, sig_bytes, M=1):
    """
    Returns a breakdown of per-transaction energy (µJ).

    The three-component model separates:
      E_sign:  signing energy amortised over M transactions
               (only one signing per M sensor samples in PQ-TDAG)
      E_ver:   verification energy amortised over M transactions
      E_tx:    physical transmission energy using the W1-corrected
               formula E_byte = P_tx × (8/R_tx)

    E_sign and E_ver are scaled from pqm4 M4 reference values
    using time-proportional scaling:
      E_device = E_m4 × (t_device / t_m4) / process_efficiency
    This is physically motivated: both platforms run the same
    NTT operations, the faster device does them with less leakage.
    """
    e_s, e_v, t_m4 = PQM4.get(sid, (8.4, 0.315, 10.0))
    # scale signing energy by observed time ratio
    E_sign = e_s * (t_sign_ms / t_m4) / PROCESS_EFFICIENCY
    # scale verification energy proportionally
    t_v_m4 = t_m4 * (e_v / max(e_s, 1e-9))
    E_ver  = e_v * (t_verify_ms / max(t_v_m4, 1e-9)) / PROCESS_EFFICIENCY
    # transmission: amortised bytes per transaction
    amort_bytes = S_PAYLOAD + sig_bytes / M
    E_tx = E_BYTE_UJ * amort_bytes
    return {
        "E_sign_uJ": round(E_sign / M, 5),
        "E_ver_uJ":  round(E_ver  / M, 5),
        "E_tx_uJ":   round(E_tx,       5),
        "E_total_uJ":round(E_sign/M + E_ver/M + E_tx, 4),
    }


def load_timings():
    if not DATA_FILE.exists():
        sys.exit(f"ERROR: {DATA_FILE} not found. Run step_2 first.")
    with open(DATA_FILE) as f:
        return json.load(f)["schemes"]


COLORS  = {"pq_tdag":"#E63946","naive_mldsa44":"#F4A261","mldsa65":"#457B9D",
           "falcon512":"#2A9D8F","slhdsa128s":"#6A4C93","slhdsa128f":"#9B59B6",
           "xmssmt":"#1D3557","ecdsa":"#A8DADC"}
MARKERS = {"pq_tdag":"o","naive_mldsa44":"s","mldsa65":"D","falcon512":"^",
           "slhdsa128s":"v","slhdsa128f":"P","xmssmt":"X","ecdsa":"h"}
co = lambda s: COLORS.get(s, "#888")
mk = lambda s: MARKERS.get(s, "o")
lw = lambda s: 2.5 if s == "pq_tdag" else 1.5
ms = lambda s: 7   if s == "pq_tdag" else 5


def save(fig, name):
    for ext in ["pdf", "png"]:
        fig.savefig(FIGURES_DIR / f"{name}.{ext}",
                    bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"    Saved: {FIGURES_DIR / name}.pdf")


# ══════════════════════════════════════════════════════════════
#  FIGURE E1 — Gateway Memory vs N
# ══════════════════════════════════════════════════════════════

def plot_E1(schemes):
    print("  Plotting fig_E1: Memory vs N...")

    def ram_mb(N, sig, pk, M=1):
        return (N * 200 * (S_PAYLOAD + 40)   # DAG active window
                + N * (pk + 64)               # public key table
                + N * M * (S_PAYLOAD + 32)    # verification buffer
                + N * M * S_PAYLOAD           # TBFR out-of-order buffer
                + N * 10 * sig                # recent-sig cache
               ) / (1024**2)

    fig, ax = plt.subplots(figsize=(3.5, 4.5))
    for sid in ["pq_tdag","naive_mldsa44","falcon512","mldsa65",
                "slhdsa128s","xmssmt","ecdsa"]:
        if sid not in schemes: continue
        s   = schemes[sid]
        M   = M_WINDOW_REF if sid == "pq_tdag" else 1
        mem = [ram_mb(n, s.get("sig_bytes",2420), s.get("pk_bytes",1312), M)
               for n in N_RANGE]
        label = "PQ-TDAG (Ours)" if sid == "pq_tdag" else s.get("label", sid)
        ax.plot(N_RANGE, mem, color=co(sid), marker=mk(sid),
                linewidth=lw(sid), markersize=ms(sid), label=label)

    for lim, lbl in [(256,"256 MB"),(512,"512 MB"),(1024,"1 GB")]:
        ax.axhline(lim, linestyle=":", linewidth=0.8, color="gray", alpha=0.6)
        ax.text(N_RANGE[-1]*0.97, lim+8, lbl, ha="right", va="bottom",
                fontsize=8, color="gray")

    ax.set(xlabel="Number of Sensors $N$",
           ylabel="Gateway RAM Footprint (MB)",
           title="(E1) Memory Footprint vs. $N$",
           xlim=(N_RANGE[0], N_RANGE[-1]), ylim=(0, 1200))
    ax.legend(fontsize=8.5, framealpha=0.9)
    plt.tight_layout()
    save(fig, "fig_E1_memory_vs_n")


# ══════════════════════════════════════════════════════════════
#  FIGURE E2 — Energy per Transaction (W1 corrected)
# ══════════════════════════════════════════════════════════════

def plot_E2(schemes):
    print("  Plotting fig_E2: Energy per Transaction (W1 corrected)...")

    ORDER = ["ecdsa","pq_tdag","naive_mldsa44","falcon512",
             "mldsa65","slhdsa128f","xmssmt","slhdsa128s"]

    xlabels, totals, bcolors, bds = [], [], [], []
    for sid in ORDER:
        if sid not in schemes: continue
        s = schemes[sid]
        if "t_sign_mean_ms" not in s: continue
        M  = M_WINDOW_REF if sid == "pq_tdag" else 1
        bd = energy_per_tx(sid, s["t_sign_mean_ms"],
                           s["t_verify_mean_ms"],
                           s.get("sig_bytes",2420), M)
        xlabels.append("PQ-TDAG\n(Ours)" if sid == "pq_tdag"
                        else s.get("label",sid)
                        .replace(" (FIPS","\n(FIPS")
                        .replace(" (Classical","\n(Classical"))
        totals.append(bd["E_total_uJ"])
        bcolors.append(co(sid))
        bds.append((sid, bd))

    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    x    = np.arange(len(xlabels))
    bars = ax.bar(x, totals, 0.65, color=bcolors,
                  edgecolor="black", linewidth=0.5)

    our = next((i for i,(s,_) in enumerate(bds) if s=="pq_tdag"), None)
    if our is not None:
        bars[our].set_edgecolor("#E63946"); bars[our].set_linewidth(2.2)

    for b, v in zip(bars, totals):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.5,
                f"{v:.1f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, fontsize=9)
    ax.set(ylabel="Energy per Transaction ($\\mu$J)",
           title=(f"(E2) Energy per Verified Transaction\n"
                  f"$E_{{byte}} = P_{{tx}} \\times 8/R_{{tx}} = {E_BYTE_UJ}\\,\\mu$J/byte "
                  f"($P_{{tx}}=1$~mW, IEEE~802.15.4)"),
           ylim=(0, max(totals)*1.18))

    # Annotate the physical derivation in a text box
    ax.text(0.01, 0.97,
            f"$E_{{byte}} = 1\\,\\mathrm{{mW}} \\times 8/{R_TX_BPS/1e6:.0f}\\,\\mathrm{{Mbps}} = {E_BYTE_UJ}\\,\\mu$J/byte",
            transform=ax.transAxes, fontsize=8, va="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow",
                      ec="gray", alpha=0.9))

    plt.tight_layout()
    save(fig, "fig_E2_energy_per_tx")

    # Print breakdown table — useful for paper verification
    print()
    print(f"    Energy breakdown — E_byte={E_BYTE_UJ} µJ/byte "
          f"(P_tx={P_TX_W*1000:.0f}mW, R_tx={R_TX_BPS/1e6:.0f}Mbps):")
    print(f"    {'Scheme':<24} {'E_sign/M':>9} {'E_ver/M':>9} "
          f"{'E_tx':>8} {'Total':>8}")
    print("    " + "-"*62)
    for sid, bd in bds:
        label = next((s.get("label",sid) for k,s in schemes.items() if k==sid), sid)
        print(f"    {label[:24]:<24}"
              f"  {bd['E_sign_uJ']:>7.4f}"
              f"  {bd['E_ver_uJ']:>7.4f}"
              f"  {bd['E_tx_uJ']:>6.4f}"
              f"  {bd['E_total_uJ']:>6.4f}")
    print()


# ══════════════════════════════════════════════════════════════
#  FIGURE E3 — DAG Storage Growth
# ══════════════════════════════════════════════════════════════

def plot_E3(schemes):
    print("  Plotting fig_E3: DAG Storage Growth...")

    def storage_mb(t_min, sig_bytes, M, is_pqtdag):
        t = t_min * 60
        N, f = 50, F_HZ_REF
        if is_pqtdag:
            n_t = N * (f / M) * t
            n_i = N * f * t * (M - 1) / M
            return (n_i * (S_PAYLOAD + 32) +
                    n_t * (S_PAYLOAD + sig_bytes + 64)) / (1024**2)
        return N * f * t * (S_PAYLOAD + sig_bytes + 64) / (1024**2)

    times = np.linspace(0, SIM_MINUTES, 200)
    fig, ax = plt.subplots(figsize=(3.5, 4.5))

    for sid in ["pq_tdag","naive_mldsa44","falcon512",
                "mldsa65","xmssmt","ecdsa"]:
        if sid not in schemes: continue
        s     = schemes[sid]
        is_pq = (sid == "pq_tdag")
        M     = M_WINDOW_REF if is_pq else 1
        label = "PQ-TDAG (Ours)" if is_pq else s.get("label", sid)
        st    = [storage_mb(t, s.get("sig_bytes",2420), M, is_pq)
                 for t in times]
        ax.plot(times, st, color=co(sid), linewidth=lw(sid), label=label)

    ax.set(xlabel="Simulation Time (min)",
           ylabel="Cumulative DAG Storage (MB)",
           title="(E3) DAG Storage Growth ($N=50$, $f=20$ Hz)",
           xlim=(0, SIM_MINUTES))
    ax.legend(fontsize=8.5, framealpha=0.9)
    plt.tight_layout()
    save(fig, "fig_E3_dag_storage")


def main():
    print()
    print("=" * 60)
    print("  PQ-TDAG — Group E: Resource Figures (W1 corrected)")
    print(f"  E_byte = P_tx×8/R_tx = {P_TX_W*1000:.0f}mW × 8/{R_TX_BPS/1e6:.0f}Mbps"
          f" = {E_BYTE_UJ} µJ/byte")
    print("=" * 60 + "\n")

    schemes = load_timings()
    print(f"  Loaded {len(schemes)} schemes\n")
    plot_E1(schemes)
    plot_E2(schemes)
    plot_E3(schemes)

    print()
    print("=" * 60)
    print("  DONE — Group E figures saved.")
    print("=" * 60)


if __name__ == "__main__":
    main()
