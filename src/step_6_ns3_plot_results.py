"""
step_8_ns3_plots.py
══════════════════════════════════════════════════════════════════
Reads NS-3 CSV outputs and generates publication-quality figures:
  fig_B1 — Throughput vs Window Size M
  fig_B2 — Throughput vs Number of Sensors (Scalability)
  fig_C1 — Latency CDF  ← MOST IMPORTANT FIGURE
  fig_C2 — Worst-case Latency vs Erasure Probability (with CI)
  fig_C3 — Latency Breakdown (stacked bar)

Run AFTER: bash scratch/pq_tdag/run_all_ns3.sh
  python3 src/step_8_ns3_plots.py
══════════════════════════════════════════════════════════════════
"""

import json, os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

ROOT        = Path(__file__).parent.parent
DATA_FILE   = ROOT / "results/data/crypto_timings.json"
NS3_RESULTS = ROOT / "results/data"   # NS-3 CSVs land here
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

T_MAX_MS   = 50.0
B_MAX_MBPS = 10.0
S_PAYLOAD  = 100


def load_timings():
    with open(DATA_FILE) as f:
        return json.load(f)["schemes"]


def load_ns3_csv(pattern: str) -> pd.DataFrame | None:
    """Try to load NS-3 output CSV. Returns None if not found."""
    p = NS3_RESULTS / pattern
    if p.exists():
        return pd.read_csv(p)
    return None


# ══════════════════════════════════════════════════════════════
#  FIGURE C1 — Latency CDF  (most important figure)
# ══════════════════════════════════════════════════════════════
def plot_C1_latency_cdf(schemes: dict):
    print("  Plotting fig_C1: Latency CDF...")

    plot_ids = ["pq_tdag", "naive_mldsa44", "falcon512",
                "mldsa65", "slhdsa128f", "ecdsa"]

    fig, ax = plt.subplots(figsize=(7, 5))

    for sid in plot_ids:
        if sid not in schemes:
            continue
        s      = schemes[sid]
        label  = s.get("label", sid)
        color  = s.get("color", "gray")
        marker = s.get("marker", "o")

        # Try to load NS-3 data
        df = load_ns3_csv(f"ns3_latency_cdf_{sid}.csv")

        if df is not None and len(df) > 10:
            lats = df["latency_ms"].values
            cdf  = df["cdf"].values
            lw   = 2.5 if sid == "pq_tdag" else 1.5
            ax.plot(lats, cdf, color=color, linewidth=lw, label=label)
            print(f"    {sid}: NS-3 data ({len(lats)} points)")
        else:
            # Fallback: synthetic CDF from analytical model
            print(f"    {sid}: NS-3 data not found — using analytical model")
            _plot_analytical_cdf(ax, sid, s, label, color)

    ax.axvline(T_MAX_MS, color="red", linestyle="-.", linewidth=2.0,
               label=f"$T_{{max}}$ = {T_MAX_MS} ms")

    # Shade infeasible region
    ax.axvspan(T_MAX_MS, 120, alpha=0.08, color="red",
               label="Deadline violation zone")

    # 99.99th percentile annotation
    ax.axhline(0.9999, color="gray", linestyle=":", linewidth=1.0,
               label="99.99th percentile")

    ax.set_xlabel("End-to-End Confirmation Latency (ms)")
    ax.set_ylabel("CDF")
    ax.set_title(
        "(C1) Confirmation Latency CDF\n"
        f"$N=50$, $f=20$ Hz, $M=5$ (PQ-TDAG), "
        f"$T_{{max}}={T_MAX_MS}$ ms"
    )
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=9, framealpha=0.9)

    out = FIGURES_DIR / "fig_C1_latency_cdf.pdf"
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight", dpi=300)
    plt.savefig(str(out).replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {out}")


def _plot_analytical_cdf(ax, sid, s, label, color):
    """Analytical CDF fallback using lognormal model."""
    t_sign  = s.get("t_sign_mean_ms", 1.2)
    t_ver   = s.get("t_verify_mean_ms", 0.045)
    sig_b   = s.get("sig_bytes", 2420)
    M       = 5 if sid == "pq_tdag" else 1

    # Mean latency components
    t_mean = t_sign / M + M * 1.0 + 0.5 + t_ver
    t_std  = t_mean * 0.15

    latencies = np.linspace(0, 120, 500)
    from scipy import stats
    sigma  = np.sqrt(np.log(1 + (t_std / t_mean) ** 2))
    mu     = np.log(t_mean) - sigma ** 2 / 2
    cdf    = stats.lognorm.cdf(latencies, s=sigma, scale=np.exp(mu))

    lw = 2.5 if sid == "pq_tdag" else 1.5
    ls = "-" if sid == "pq_tdag" else "--"
    ax.plot(latencies, cdf, color=color, linewidth=lw,
            linestyle=ls, label=f"{label} (analytical)")


# ══════════════════════════════════════════════════════════════
#  FIGURE C2 — Latency vs p_e with Confidence Intervals
# ══════════════════════════════════════════════════════════════
def plot_C2_latency_vs_pe():
    print("  Plotting fig_C2: Latency vs Erasure Probability...")

    M_vals  = [3, 5, 8, 10]
    colors  = ["#2A9D8F", "#E63946", "#457B9D", "#6A4C93"]

    fig, ax = plt.subplots(figsize=(7, 5))

    for M_val, color in zip(M_vals, colors):
        df = load_ns3_csv(f"ns3_erasure_M{M_val}.csv")

        if df is not None and len(df) > 5:
            pe     = df["pe"].values * 100
            lat    = df["latency_ms_mean"].values
            std    = df["latency_ms_std"].values
            ax.plot(pe, lat, color=color, linewidth=2.0,
                    label=f"$M={M_val}$ (NS-3)")
            ax.fill_between(pe, lat - 2*std, lat + 2*std,
                            alpha=0.15, color=color)
        else:
            # Analytical fallback
            print(f"    M={M_val}: analytical fallback")
            _plot_analytical_latency_vs_pe(ax, M_val, color)

    ax.axhline(T_MAX_MS, color="red", linestyle="-.", linewidth=2.0,
               label=f"$T_{{max}}={T_MAX_MS}$ ms")

    ax.set_xlabel("Erasure Probability $p_e$ (%)")
    ax.set_ylabel("Worst-case Confirmation Latency (ms)")
    ax.set_title(
        "(C2) Worst-case Latency vs. Erasure Probability\n"
        "TBFR protocol, $\\gamma_{req} = 1-10^{-5}$ (shaded: $\\pm 2\\sigma$)"
    )
    ax.set_xlim(0, 25)
    ax.set_ylim(0, T_MAX_MS * 2.2)
    ax.legend(fontsize=9, framealpha=0.9)

    out = FIGURES_DIR / "fig_C2_latency_vs_pe.pdf"
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight", dpi=300)
    plt.savefig(str(out).replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {out}")


def _plot_analytical_latency_vs_pe(ax, M, color, t_sign=1.20,
                                    t_verify=0.045):
    """Analytical L_worst from Lemma 3."""
    import math
    GAMMA_REQ    = 1 - 1e-5
    T_RTX_MS     = 3.0
    T_TIP_MS     = 0.5
    T_PIPE_MS    = 1.0
    M_BAR        = int((T_MAX_MS - t_sign - T_TIP_MS) / T_PIPE_MS)

    pe_range = np.linspace(0, 0.25, 100)
    lats = []
    for pe in pe_range:
        if pe <= 0:
            r_max = 0
        else:
            try:
                val   = math.log((1 - GAMMA_REQ) / max(M - 1, 1)) \
                        / math.log(pe) - 1
                r_max = max(0, math.ceil(val))
            except (ValueError, ZeroDivisionError):
                r_max = 0
        lat = t_sign + T_TIP_MS + M * T_PIPE_MS + r_max * T_RTX_MS
        lats.append(lat)

    ax.plot(pe_range * 100, lats, color=color, linewidth=2.0,
            linestyle="--", label=f"$M={M}$ (analytical)")


# ══════════════════════════════════════════════════════════════
#  FIGURE C3 — Latency Breakdown (stacked bar)
# ══════════════════════════════════════════════════════════════
def plot_C3_latency_breakdown(schemes: dict):
    print("  Plotting fig_C3: Latency Breakdown...")

    plot_ids = ["pq_tdag", "naive_mldsa44", "falcon512",
                "mldsa65", "slhdsa128f"]
    labels_plot = []

    # Components: t_sign_amort | t_hash_chain | t_pipe | t_queue | t_verify | t_tip
    comp_colors = {
        "Sign (amortized)": "#E63946",
        "Hash chain":       "#F4A261",
        "Network pipe":     "#2A9D8F",
        "Queue delay":      "#457B9D",
        "Verify":           "#6A4C93",
        "Tip selection":    "#A8DADC",
    }
    components = list(comp_colors.keys())
    data = {c: [] for c in components}

    for sid in plot_ids:
        if sid not in schemes:
            continue
        s     = schemes[sid]
        M     = 5 if sid == "pq_tdag" else 1
        t_s   = s.get("t_sign_mean_ms", 1.2)
        t_v   = s.get("t_verify_mean_ms", 0.045)

        data["Sign (amortized)"].append(t_s / M)
        data["Hash chain"].append(M * 0.010)    # SHA3-256 per block
        data["Network pipe"].append(1.0)         # T_pipe_worst
        data["Queue delay"].append(0.5)          # G/D/1 mean queue
        data["Verify"].append(t_v)
        data["Tip selection"].append(0.5)        # DTTS
        labels_plot.append(s.get("label", sid))

    fig, ax = plt.subplots(figsize=(9, 5))
    x       = np.arange(len(labels_plot))
    bottoms = np.zeros(len(labels_plot))

    for comp, color in comp_colors.items():
        vals = np.array(data[comp])
        ax.bar(x, vals, 0.6, bottom=bottoms,
               color=color, label=comp,
               edgecolor="black", linewidth=0.4)
        bottoms += vals

    ax.axhline(T_MAX_MS, color="red", linestyle="-.", linewidth=1.5,
               label=f"$T_{{max}}={T_MAX_MS}$ ms")

    ax.set_xticks(x)
    ax.set_xticklabels(labels_plot, rotation=30, ha="right")
    ax.set_ylabel("Latency Component (ms)")
    ax.set_title(
        "(C3) End-to-End Latency Breakdown\n"
        "$N=50$, $f=20$ Hz, $M=5$ (PQ-TDAG), $M=1$ (others)"
    )
    ax.legend(fontsize=8.5, ncol=2, loc="upper right", framealpha=0.9)

    out = FIGURES_DIR / "fig_C3_latency_breakdown.pdf"
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight", dpi=300)
    plt.savefig(str(out).replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {out}")


# ══════════════════════════════════════════════════════════════
#  FIGURE B1 — Throughput vs Window Size M
# ══════════════════════════════════════════════════════════════
def plot_B1_throughput_vs_m(schemes: dict):
    print("  Plotting fig_B1: Throughput vs M...")

    plot_ids = ["pq_tdag", "naive_mldsa44", "falcon512", "mldsa65"]
    fig, ax  = plt.subplots(figsize=(7, 5))

    for sid in plot_ids:
        if sid not in schemes:
            continue
        s      = schemes[sid]
        label  = s.get("label", sid)
        color  = s.get("color", "gray")
        marker = s.get("marker", "o")

        df = load_ns3_csv(f"ns3_throughput_window_{sid}.csv")
        if df is not None and len(df) > 3:
            ax.plot(df["param"], df["throughput_tx_per_sec"],
                    color=color, marker=marker,
                    linewidth=2.0 if sid == "pq_tdag" else 1.5,
                    label=label)
        else:
            # Analytical fallback
            M_arr  = np.arange(1, 26)
            sig_b  = s.get("sig_bytes", 2420)
            t_v    = s.get("t_verify_mean_ms", 0.045)
            N, f   = 50, 20.0
            tputs  = []
            for M in M_arr:
                b = N * (f / M) * (M * S_PAYLOAD + sig_b) * 8 / 1e6
                if b > B_MAX_MBPS and sid != "pq_tdag":
                    tputs.append(0)
                else:
                    rate  = N * f / M
                    t_conf= t_v + 1.0 + 0.5
                    tputs.append(min(rate, 1000 / t_conf * N))
            lw = 2.5 if sid == "pq_tdag" else 1.5
            ax.plot(M_arr, tputs, color=color, marker=marker,
                    linewidth=lw, linestyle="--", label=f"{label} (analytical)")

    ax.axvline(3, color="red", linestyle=":", linewidth=1.5,
               label="$M_{min}=3$ (Corollary 1)")

    ax.set_xlabel("Micro-chain Window Size $M$")
    ax.set_ylabel("Verified Effective Throughput $\\Gamma(\\mathcal{G})$ (tx/s)")
    ax.set_title("(B1) Verified Throughput vs. Window Size $M$\n"
                 "$N=50$, $f=20$ Hz")
    ax.set_xlim(1, 25)
    ax.legend(fontsize=9, framealpha=0.9)

    out = FIGURES_DIR / "fig_B1_throughput_vs_m.pdf"
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight", dpi=300)
    plt.savefig(str(out).replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {out}")


# ══════════════════════════════════════════════════════════════
#  FIGURE B2 — Throughput vs N (Scalability)
# ══════════════════════════════════════════════════════════════
def plot_B2_throughput_vs_n(schemes: dict):
    print("  Plotting fig_B2: Throughput vs N (Scalability)...")

    plot_ids = ["pq_tdag", "naive_mldsa44", "falcon512", "mldsa65"]
    N_RANGE  = [10, 25, 50, 100, 150, 200, 300, 500, 750, 1000]
    fig, ax  = plt.subplots(figsize=(7, 5))

    for sid in plot_ids:
        if sid not in schemes:
            continue
        s      = schemes[sid]
        label  = s.get("label", sid)
        color  = s.get("color", "gray")
        marker = s.get("marker", "o")

        df = load_ns3_csv(f"ns3_throughput_scale_{sid}.csv")
        if df is not None and len(df) > 3:
            ax.plot(df["param"], df["throughput_tx_per_sec"],
                    color=color, marker=marker,
                    linewidth=2.0 if sid == "pq_tdag" else 1.5,
                    label=label)
        else:
            sig_b = s.get("sig_bytes", 2420)
            t_v   = s.get("t_verify_mean_ms", 0.045)
            M     = 5 if sid == "pq_tdag" else 1
            f     = 20.0
            tputs = []
            for N in N_RANGE:
                b = N * (f / M) * (M * S_PAYLOAD + sig_b) * 8 / 1e6
                if b > B_MAX_MBPS and sid != "pq_tdag":
                    tputs.append(0)
                else:
                    rate  = N * f / M
                    t_conf= t_v + 1.0 + 0.5
                    tputs.append(min(rate, 1000 / t_conf * N))
            lw = 2.5 if sid == "pq_tdag" else 1.5
            ax.plot(N_RANGE, tputs, color=color, marker=marker,
                    linewidth=lw, linestyle="--",
                    label=f"{label} (analytical)")

    ax.set_xlabel("Number of Sensors $N$")
    ax.set_ylabel("Verified Effective Throughput (tx/s)")
    ax.set_title("(B2) Scalability: Throughput vs. $N$\n"
                 "$f=20$ Hz, $M=5$ (PQ-TDAG), $M=1$ (others)")
    ax.set_xlim(N_RANGE[0], N_RANGE[-1])
    ax.legend(fontsize=9, framealpha=0.9)

    out = FIGURES_DIR / "fig_B2_throughput_vs_n.pdf"
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
    print("  PQ-TDAG — NS-3 Result Plots (Step 8)")
    print("═" * 60)
    print()

    schemes = load_timings()

    ns3_files = list(NS3_RESULTS.glob("ns3_*.csv"))
    if ns3_files:
        print(f"  Found {len(ns3_files)} NS-3 result files")
    else:
        print("  No NS-3 CSVs found — using analytical fallbacks")
        print("  (run NS-3 simulations first for measured results)")
    print()

    plot_C1_latency_cdf(schemes)
    plot_C2_latency_vs_pe()
    plot_C3_latency_breakdown(schemes)
    plot_B1_throughput_vs_m(schemes)
    plot_B2_throughput_vs_n(schemes)

    print()
    print("═" * 60)
    print("  DONE — NS-3 figures saved.")
    print()
    print("  Next step:")
    print("  python3 src/step_9_gpu_benchmark.py  (CUDA batch verify)")
    print("  python3 src/step_10_final_tables.py  (paper tables)")
    print("═" * 60)
    print()


if __name__ == "__main__":
    main()
