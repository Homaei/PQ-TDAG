"""
step_3_group_A_bandwidth_plots.py
══════════════════════════════════════════════════════════════════
Group A: Bandwidth & Feasibility Figures
  fig_A1 — Required BW vs Sampling Frequency  (all 8 schemes)
  fig_A2 — Required BW vs Number of Sensors   (all 8 schemes)
  fig_A3 — M_min vs B_max                     (PQC schemes)

Reads:  results/data/crypto_timings.json  (from Step 2)
Output: results/figures/fig_A*.pdf

Run:
  python3 src/step_3_group_A_bandwidth_plots.py
══════════════════════════════════════════════════════════════════
"""

import json, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────
ROOT        = Path(__file__).parent.parent
DATA_FILE   = ROOT / "results/data/crypto_timings.json"
FIGURES_DIR = ROOT / "results/figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ── Publication style ────────────────────────────────────────
plt.rcParams.update({
    "font.family":        "serif",
    "font.size":          11,
    "axes.labelsize":     12,
    "axes.titlesize":     12,
    "legend.fontsize":    9,
    "xtick.labelsize":    10,
    "ytick.labelsize":    10,
    "figure.dpi":         150,
    "axes.grid":          True,
    "grid.alpha":         0.35,
    "grid.linestyle":     "--",
    "lines.linewidth":    1.8,
    "lines.markersize":   6,
})

# ── ICS reference parameters ─────────────────────────────────
B_MAX_MBPS      = 10.0      # 5G URLLC cap
S_PAYLOAD_B     = 100       # ICS sensor payload
N_REF           = 50        # reference sensor count
F_REF_HZ        = 20        # reference sampling frequency
M_WINDOW_REF    = 5         # reference window size (PQ-TDAG)

# ── Sweep ranges ─────────────────────────────────────────────
FREQ_RANGE = np.array([1, 2, 4, 5, 8, 10, 15, 20, 30,
                        40, 50, 60, 80, 100, 120])
N_RANGE    = np.array([10, 25, 50, 100, 150, 200,
                        300, 500, 750, 1000])
BMAX_RANGE = np.linspace(1, 25, 200)


# ══════════════════════════════════════════════════════════════
#  LOAD MEASURED TIMINGS
# ══════════════════════════════════════════════════════════════
def load_timings() -> dict:
    if not DATA_FILE.exists():
        print(f"ERROR: {DATA_FILE} not found.")
        print("Run Step 2 first: python3 src/step_2_crypto_benchmark.py")
        sys.exit(1)
    with open(DATA_FILE) as f:
        data = json.load(f)
    return data["schemes"]


# ══════════════════════════════════════════════════════════════
#  BANDWIDTH FORMULAS (from paper equations)
# ══════════════════════════════════════════════════════════════
def bw_naive_mbps(N, f, sig_bytes):
    """Eq. 1 — naive (every packet signed individually)."""
    return N * f * (S_PAYLOAD_B + sig_bytes) * 8 / 1e6


def bw_pqtdag_mbps(N, f, sig_bytes, M):
    """Eq. B_red(M) — PQ-TDAG with micro-chain window M."""
    return N * (f / M) * (M * S_PAYLOAD_B + sig_bytes) * 8 / 1e6


def m_min(sig_bytes, N, f, B_max_mbps):
    """Corollary 1 — minimum window size for bandwidth feasibility."""
    denom = (B_max_mbps * 1e6) / (N * f * 8) - S_PAYLOAD_B
    if denom <= 0:
        return np.inf
    return np.ceil(sig_bytes / denom)


# ══════════════════════════════════════════════════════════════
#  FIGURE A1 — BW vs Sampling Frequency
# ══════════════════════════════════════════════════════════════
def plot_A1_bw_vs_freq(schemes: dict):
    print("  Plotting fig_A1: BW vs Sampling Frequency...")

    fig, ax = plt.subplots(figsize=(7, 5))

    for sid, sdata in schemes.items():
        sig_b  = sdata.get("sig_bytes", 0)
        label  = sdata.get("label", sid)
        color  = sdata.get("color", "gray")
        marker = sdata.get("marker", "o")
        role   = sdata.get("role", "baseline")

        if sid == "pq_tdag":
            # PQ-TDAG — show M=5 (reference) and M=3 (minimum)
            bw5 = [bw_pqtdag_mbps(N_REF, f, sig_b, 5) for f in FREQ_RANGE]
            bw3 = [bw_pqtdag_mbps(N_REF, f, sig_b, 3) for f in FREQ_RANGE]
            ax.plot(FREQ_RANGE, bw5, color=color, marker=marker,
                    linewidth=2.5, markersize=7, zorder=5,
                    label=f"PQ-TDAG M=5 (Ours)")
            ax.plot(FREQ_RANGE, bw3, color=color, marker=marker,
                    linewidth=1.5, linestyle=":", markersize=5, zorder=5,
                    label=f"PQ-TDAG M=3 (M_min)")
        elif sid == "naive_mldsa44":
            # Skip naive in this plot — it's same line as others with sig=2420
            # Already shown as Dilithium2 baseline
            bw = [bw_naive_mbps(N_REF, f, sig_b) for f in FREQ_RANGE]
            ax.plot(FREQ_RANGE, bw, color=color, marker=marker,
                    linewidth=1.8, markersize=5, linestyle="--",
                    label=label)
        else:
            bw = [bw_naive_mbps(N_REF, f, sig_b) for f in FREQ_RANGE]
            ls = "-" if role == "proposed" else "-"
            ax.plot(FREQ_RANGE, bw, color=color, marker=marker,
                    linewidth=1.5, markersize=5, linestyle=ls,
                    label=label)

    # B_max line
    ax.axhline(B_MAX_MBPS, color="black", linestyle="-.",
               linewidth=1.8, label=f"$B_{{max}}$ = {B_MAX_MBPS} Mbps (5G URLLC)")

    # Collapse annotation
    ax.annotate("← Throughput Collapse", xy=(4, B_MAX_MBPS + 0.3),
                fontsize=8.5, color="black",
                arrowprops=dict(arrowstyle="->", color="black"),
                xytext=(8, 14))

    ax.set_xlabel("Sensor Sampling Frequency $f$ (Hz)")
    ax.set_ylabel("Required Bandwidth $B_{req}$ (Mbps)")
    ax.set_title(
        f"(A1) Required Bandwidth vs. Sampling Frequency\n"
        f"$N={N_REF}$ sensors, $S_{{payload}}={S_PAYLOAD_B}$ B"
    )
    ax.set_xlim(FREQ_RANGE[0], FREQ_RANGE[-1])
    ax.set_ylim(0, 80)
    ax.legend(loc="upper left", fontsize=8.5, ncol=1,
              framealpha=0.9)

    out = FIGURES_DIR / "fig_A1_bw_vs_freq.pdf"
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight", dpi=300)
    plt.savefig(str(out).replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {out}")


# ══════════════════════════════════════════════════════════════
#  FIGURE A2 — BW vs Number of Sensors
# ══════════════════════════════════════════════════════════════
def plot_A2_bw_vs_n(schemes: dict):
    print("  Plotting fig_A2: BW vs Number of Sensors...")

    fig, ax = plt.subplots(figsize=(7, 5))

    for sid, sdata in schemes.items():
        sig_b  = sdata.get("sig_bytes", 0)
        label  = sdata.get("label", sid)
        color  = sdata.get("color", "gray")
        marker = sdata.get("marker", "o")

        if sid == "pq_tdag":
            bw = [bw_pqtdag_mbps(n, F_REF_HZ, sig_b, M_WINDOW_REF) for n in N_RANGE]
            ax.plot(N_RANGE, bw, color=color, marker=marker,
                    linewidth=2.5, markersize=7, zorder=5,
                    label=f"PQ-TDAG M={M_WINDOW_REF} (Ours)")
        elif sid == "naive_mldsa44":
            bw = [bw_naive_mbps(n, F_REF_HZ, sig_b) for n in N_RANGE]
            ax.plot(N_RANGE, bw, color=color, marker=marker,
                    linewidth=1.8, linestyle="--", markersize=5,
                    label=label)
        else:
            bw = [bw_naive_mbps(n, F_REF_HZ, sig_b) for n in N_RANGE]
            ax.plot(N_RANGE, bw, color=color, marker=marker,
                    linewidth=1.5, markersize=5, label=label)

    ax.axhline(B_MAX_MBPS, color="black", linestyle="-.",
               linewidth=1.8, label=f"$B_{{max}}$ = {B_MAX_MBPS} Mbps")

    # Feasibility region shading
    ax.fill_between(N_RANGE, 0, B_MAX_MBPS,
                    alpha=0.08, color="green", label="Feasible region")

    ax.set_xlabel("Number of Sensors $N$")
    ax.set_ylabel("Required Bandwidth $B_{req}$ (Mbps)")
    ax.set_title(
        f"(A2) Required Bandwidth vs. Number of Sensors\n"
        f"$f={F_REF_HZ}$ Hz, $S_{{payload}}={S_PAYLOAD_B}$ B"
    )
    ax.set_xlim(N_RANGE[0], N_RANGE[-1])
    ax.set_ylim(0, 100)
    ax.legend(loc="upper left", fontsize=8.5, ncol=1, framealpha=0.9)

    out = FIGURES_DIR / "fig_A2_bw_vs_n.pdf"
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight", dpi=300)
    plt.savefig(str(out).replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {out}")


# ══════════════════════════════════════════════════════════════
#  FIGURE A3 — M_min vs B_max (Corollary 1 visualization)
# ══════════════════════════════════════════════════════════════
def plot_A3_mmin_vs_bmax(schemes: dict):
    print("  Plotting fig_A3: M_min vs B_max (Corollary 1)...")

    # Only PQC schemes with sig_bytes > ECDSA
    pqc_schemes = {
        k: v for k, v in schemes.items()
        if v.get("is_pqc", False) and k != "naive_mldsa44"
    }

    fig, ax = plt.subplots(figsize=(7, 5))

    for sid, sdata in pqc_schemes.items():
        sig_b  = sdata.get("sig_bytes", 0)
        label  = sdata.get("label", sid)
        color  = sdata.get("color", "gray")
        marker = sdata.get("marker", "o")

        mmin_vals = []
        for bmax in BMAX_RANGE:
            m = m_min(sig_b, N_REF, F_REF_HZ, bmax)
            mmin_vals.append(min(m, 50) if np.isfinite(m) else 50)

        lw = 2.5 if sid == "pq_tdag" else 1.5
        ax.plot(BMAX_RANGE, mmin_vals, color=color,
                linewidth=lw, label=label)

    # Reference lines
    ax.axvline(B_MAX_MBPS, color="black", linestyle="-.",
               linewidth=1.5, label=f"$B_{{max}}$ = {B_MAX_MBPS} Mbps")
    ax.axhline(3, color="red", linestyle="--",
               linewidth=1.2, alpha=0.7, label="$M_{{min}}=3$ (our reference)")

    # Annotate operating point
    ax.annotate(f"Operating point\n$M_{{min}}=3, B_{{max}}=10$ Mbps",
                xy=(10, 3), xytext=(14, 8),
                fontsize=8.5, color="red",
                arrowprops=dict(arrowstyle="->", color="red"))

    ax.set_xlabel("Channel Capacity $B_{max}$ (Mbps)")
    ax.set_ylabel("Minimum Window Size $M_{min}$")
    ax.set_title(
        "(A3) Minimum Window Size vs. Channel Capacity\n"
        "(Corollary 1 — Bandwidth Feasibility Condition)"
    )
    ax.set_xlim(1, 25)
    ax.set_ylim(0, 30)
    ax.legend(fontsize=8.5, ncol=1, framealpha=0.9)

    out = FIGURES_DIR / "fig_A3_mmin_vs_bmax.pdf"
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
    print("  PQ-TDAG — Group A: Bandwidth Figures (Step 3)")
    print("═" * 60)
    print()

    schemes = load_timings()
    print(f"  Loaded {len(schemes)} schemes from crypto_timings.json")
    print()

    plot_A1_bw_vs_freq(schemes)
    plot_A2_bw_vs_n(schemes)
    plot_A3_mmin_vs_bmax(schemes)

    print()
    print("═" * 60)
    print("  DONE — Group A figures saved.")
    print()
    print("  Files:")
    for f in sorted(FIGURES_DIR.glob("fig_A*.pdf")):
        print(f"    {f}")
    print()
    print("  Next step:")
    print("  python3 src/step_4_group_E_resources.py")
    print("═" * 60)
    print()


if __name__ == "__main__":
    main()
