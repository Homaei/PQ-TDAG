"""
src/step_5_security_figures.py
───────────────────────────────────────────────────────────────────
Group D: Security Figures
  fig_D1 — Byzantine Robustness (extended to n=3,5,7,9 per W6 fix)
  fig_D2 — TBFR vs Standard ARQ Recovery
  fig_D3 — Attack Success Probability vs M (Theorem 1)
  fig_D4 — W6 fix: Byzantine threshold table for different n values

W6 Fix:
  The original evaluation used n=9 gateways (|F|_max=2).
  A reviewer correctly noted that real ICS deployments often
  use n=3 gateways, where the Byzantine threshold degrades to
  |F|_max=0 — effectively CFT (Crash-Fault Tolerant) rather
  than BFT (Byzantine-Fault Tolerant).

  We address this by:
  1. Extending the Byzantine simulation to n ∈ {3, 5, 7, 9}
  2. Showing that n≥4 is required for any Byzantine tolerance
  3. Adding a deployment recommendation table to the paper

  The key result: for n=3, PQ-TDAG is CFT-only, and operators
  requiring BFT must deploy n≥7 gateways (tolerates 2 faulty nodes).
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

ROOT        = Path(__file__).parent.parent
DATA_FILE   = ROOT / "results/data/crypto_timings.json"
FIGURES_DIR = ROOT / "results/figures"
LOGS_DIR    = ROOT / "results/logs"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "font.size": 11,
    "axes.labelsize": 12, "axes.titlesize": 11,
    "legend.fontsize": 9, "xtick.labelsize": 10,
    "ytick.labelsize": 10, "figure.dpi": 150,
    "axes.grid": True, "grid.alpha": 0.35, "grid.linestyle": "--",
    "lines.linewidth": 1.8, "lines.markersize": 6,
})

T_MAX_MS    = 50.0
T_RTX_MS    = 3.0
T_SIGN_MS   = 0.0436
T_TIP_MS    = 0.5
GAMMA_REQ   = 1 - 1e-5
BYZ_RATIOS  = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.33]
N_MONTE_CARLO = 200
M_RANGE     = list(range(1, 21))
PE_RANGE    = np.linspace(0.0, 0.25, 40)
rng         = np.random.default_rng(42)

# W6: gateway cluster sizes to evaluate
# n=3: minimal deployment (many small ICS cells)
# n=5: small deployment
# n=7: minimum for 2-fault BFT
# n=9: our primary evaluation (as in paper)
GATEWAY_SIZES = [3, 5, 7, 9]


def co(sid):
    COLORS = {"pq_tdag":"#E63946","naive_mldsa44":"#F4A261","mldsa65":"#457B9D",
              "falcon512":"#2A9D8F","slhdsa128s":"#6A4C93","slhdsa128f":"#9B59B6",
              "xmssmt":"#1D3557","ecdsa":"#A8DADC"}
    return COLORS.get(sid, "#888")


def save(fig, name):
    for ext in ["pdf", "png"]:
        fig.savefig(FIGURES_DIR / f"{name}.{ext}",
                    bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"    Saved: {FIGURES_DIR / name}.pdf")


# ══════════════════════════════════════════════════════════════
#  W6: Byzantine simulation for multiple n values
# ══════════════════════════════════════════════════════════════

def byzantine_threshold(n: int) -> int:
    """
    BFT threshold: floor((n-1)/3)
    This is the classic Castro-Liskov PBFT bound.
    For n=3: floor(2/3) = 0 → CFT only, not BFT
    For n=4: floor(3/3) = 1 → tolerates 1 Byzantine
    For n=7: floor(6/3) = 2 → tolerates 2 Byzantine
    """
    return (n - 1) // 3


def simulate_dag_integrity(n_gateways: int, byz_ratio: float,
                            n_sensors: int = 50, sim_steps: int = 300) -> dict:
    """
    Simulates DAG integrity under Byzantine attacks for a given
    gateway cluster size n and Byzantine ratio.

    Returns integrity rate, equivocation detection time, and
    whether the cluster is operating in BFT or CFT mode.
    """
    n_byzantine = int(n_gateways * byz_ratio)
    byz_thresh  = byzantine_threshold(n_gateways)
    is_bft      = (n_byzantine < byz_thresh) and (byz_thresh > 0)
    is_cft      = (n_byzantine == 0) and (byz_thresh == 0)

    integrity_rates  = []
    detect_times_ms  = []

    for run in range(N_MONTE_CARLO):
        rng_local  = np.random.default_rng(run * 100 + n_gateways)
        total_txs  = 0
        valid_txs  = 0

        for step in range(sim_steps):
            if n_byzantine > 0 and rng_local.random() < 0.15:
                attack = rng_local.integers(3)
                if attack == 0:
                    # Equivocation
                    if is_bft:
                        detect_t = T_RTX_MS + rng_local.exponential(1.0)
                        detect_times_ms.append(detect_t)
                    total_txs += 1
                elif attack == 1:
                    # Selective drop
                    detect_t = 1.0 + rng_local.exponential(0.5)
                    detect_times_ms.append(detect_t)
                    total_txs += 1
                else:
                    total_txs += 1
            else:
                total_txs += 1
                valid_txs += 1

        if total_txs > 0:
            integrity_rates.append(valid_txs / total_txs)

    return {
        "n_gateways":     n_gateways,
        "byz_ratio":      byz_ratio,
        "n_byzantine":    n_byzantine,
        "byz_threshold":  byz_thresh,
        "is_bft":         is_bft,
        "is_cft":         is_cft,
        "integrity_mean": float(np.mean(integrity_rates)) if integrity_rates else 0.0,
        "integrity_std":  float(np.std(integrity_rates))  if integrity_rates else 0.0,
        "detect_ms_mean": float(np.mean(detect_times_ms)) if detect_times_ms else 0.0,
        "mode":           "BFT" if (is_bft or byz_thresh > 0) else "CFT",
    }


def plot_D1_byzantine(n_gateways: int = 9):
    """Byzantine robustness for the primary n=9 evaluation."""
    print(f"  Plotting fig_D1: Byzantine Robustness (n={n_gateways})...")

    results = []
    for ratio in tqdm(BYZ_RATIOS, desc=f"    n={n_gateways} Byzantine"):
        results.append(simulate_dag_integrity(n_gateways, ratio))

    ratios_pct  = [r["byz_ratio"] * 100 for r in results]
    integ_means = [r["integrity_mean"] * 100 for r in results]
    integ_stds  = [r["integrity_std"] * 100 for r in results]
    detect_ms   = [r["detect_ms_mean"] for r in results]
    thresh_pct  = 100.0 / 3.0   # n/3 threshold in %

    bar_colors = ["#2A9D8F" if r["byz_ratio"] < 1/3
                  else "#E63946" for r in results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    ax1.bar(ratios_pct, integ_means, 4.5, color=bar_colors,
            edgecolor="black", linewidth=0.5,
            yerr=integ_stds, capsize=4,
            error_kw={"linewidth": 1.0})
    ax1.axvline(thresh_pct, color="#E63946", linestyle="--", linewidth=1.8,
                label=f"BFT threshold $n/3 = {thresh_pct:.1f}\\%$")
    ax1.fill_betweenx([80,102], 0, thresh_pct, alpha=0.05, color="#2A9D8F")
    ax1.fill_betweenx([80,102], thresh_pct, 36, alpha=0.05, color="#E63946")
    ax1.set(xlabel="Byzantine Gateway Ratio (\\%)",
            ylabel="DAG Integrity Rate (\\%)",
            title=f"(D1a) DAG Integrity ($n={n_gateways}$ gateways)",
            xlim=(-1, 36), ylim=(80, 102))
    ax1.legend(fontsize=9)

    ax2.plot(ratios_pct, detect_ms, color="#E63946", marker="o",
             linewidth=2.0, markersize=7)
    ax2.axvline(thresh_pct, color="#E63946", linestyle="--", linewidth=1.5)
    ax2.axhline(T_MAX_MS, color="#888", linestyle="-.", linewidth=1.0,
                label=f"$T_{{\\max}}={T_MAX_MS}$~ms")
    ax2.set(xlabel="Byzantine Gateway Ratio (\\%)",
            ylabel="Detection Time (ms)",
            title="(D1b) Equivocation Detection",
            xlim=(-1, 36))
    ax2.legend(fontsize=9)

    plt.tight_layout()
    save(fig, "fig_D1_byzantine_robustness")

    with open(LOGS_DIR / "byzantine_results.json", "w") as f:
        json.dump({str(r["byz_ratio"]): r for r in results}, f, indent=2)

    return results


def plot_D4_byzantine_vs_n():
    """
    W6 fix: Byzantine fault tolerance across gateway cluster sizes.

    This figure directly answers the reviewer's concern about n=3
    deployments. It shows:
      - BFT threshold |F|_max for each n
      - Which mode the system operates in (BFT vs CFT)
      - Deployment recommendation for plant operators
    """
    print("  Plotting fig_D4: Byzantine threshold vs n (W6 fix)...")

    # Simulate integrity at the n/3 threshold for each cluster size
    # (this is the operating point where BFT is just barely maintained)
    cluster_data = []
    for n in GATEWAY_SIZES:
        thresh    = byzantine_threshold(n)
        ratio_at_thresh = thresh / n if thresh > 0 else 0.0
        # Simulate at threshold - 1 (within BFT), at threshold, and at threshold + 1
        rows = []
        for n_byz in range(0, min(n, thresh + 2)):
            ratio = n_byz / n
            r     = simulate_dag_integrity(n, ratio)
            rows.append(r)
        cluster_data.append({"n": n, "threshold": thresh, "rows": rows})

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.suptitle(
        "W6: Byzantine Fault Tolerance vs. Gateway Cluster Size\n"
        "Deployment recommendations for ICS operators",
        fontsize=10, y=1.02
    )

    CLUSTER_COLORS = ["#E63946", "#F4A261", "#2A9D8F", "#457B9D"]

    # Panel (a): Integrity vs Byzantine count for each n
    ax = axes[0]
    for cd, color in zip(cluster_data, CLUSTER_COLORS):
        n      = cd["n"]
        thresh = cd["threshold"]
        rows   = cd["rows"]
        n_byz_vals = [r["n_byzantine"] for r in rows]
        integ  = [r["integrity_mean"] * 100 for r in rows]
        mode   = "BFT" if thresh > 0 else "CFT-only"
        ax.plot(n_byz_vals, integ, color=color, marker="o",
                linewidth=2.0, markersize=7,
                label=f"$n={n}$ (|F|_max={thresh}, {mode})")
        if thresh > 0:
            ax.axvline(thresh + 0.5, color=color, linestyle=":",
                       linewidth=0.8, alpha=0.5)

    ax.set(xlabel="Number of Byzantine Gateways $|\\mathcal{F}|$",
           ylabel="DAG Integrity Rate (\\%)",
           title="(D4a) Integrity vs. Byzantine Count",
           ylim=(78, 102))
    ax.legend(fontsize=8.5)

    # Panel (b): Deployment recommendation table as heatmap
    ax = axes[1]
    ax.axis("off")

    table_data = []
    col_labels = ["$n$", "|F|$_{max}$", "Mode", "Min N gateways\nfor BFT", "Recommendation"]
    for cd in cluster_data:
        n     = cd["n"]
        thresh= cd["threshold"]
        mode  = "BFT" if thresh > 0 else "CFT only"
        min_n = "n≥4" if n == 3 else ("n≥7" if thresh < 2 else "✓ sufficient")
        rec   = ("Upgrade to n≥4" if n == 3 else
                 ("Upgrade to n≥7\nfor 2-fault BFT" if thresh < 2 else
                  "Recommended"))
        table_data.append([str(n), str(thresh), mode, min_n, rec])

    table = ax.table(
        cellText=table_data,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
        bbox=[0, 0, 1, 1]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)

    # Color rows: red for CFT-only, yellow for limited BFT, green for good
    for i, cd in enumerate(cluster_data):
        thresh = cd["threshold"]
        color  = ("#ffcccc" if thresh == 0 else
                  "#fff3cd" if thresh < 2 else "#d4edda")
        for j in range(len(col_labels)):
            table[i+1, j].set_facecolor(color)

    ax.set_title("(D4b) Deployment Recommendations", fontsize=11, pad=15)

    plt.tight_layout()
    save(fig, "fig_D4_byzantine_vs_n")

    # Print the key finding for paper text
    print()
    print("    W6 Key Numbers for paper §5.5:")
    for cd in cluster_data:
        n, thresh = cd["n"], cd["threshold"]
        mode = "BFT" if thresh > 0 else "CFT-only"
        print(f"    n={n}: |F|_max={thresh}, mode={mode}")
    print()
    print("    Paper text (§5.5, Byzantine section):")
    print('    "While our evaluation uses n=9 gateways (|F|_max=2),')
    print('     constrained ICS cells often deploy n=3 gateways.')
    print('     For n=3, the BFT threshold evaluates to |F|_max=0:')
    print('     PQ-TDAG operates in CFT mode, not BFT mode.')
    print('     Plant operators requiring strict BFT must mandate')
    print('     n≥4 gateways (1-fault tolerance) or n≥7 (2-fault)."')


# ══════════════════════════════════════════════════════════════
#  FIGURE D2 — TBFR vs ARQ
# ══════════════════════════════════════════════════════════════

def plot_D2_tbfr():
    print("  Plotting fig_D2: TBFR vs ARQ...")
    import math

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))
    M_vals   = [3, 5, 8]
    colors_m = ["#2A9D8F", "#E63946", "#457B9D"]

    for M, color in zip(M_vals, colors_m):
        tbfr_t, arq_t, rates = [], [], []
        for pe in PE_RANGE:
            if pe <= 0:
                r = 0
            else:
                try:
                    v = math.log(1e-5 / max(M-1, 1)) / math.log(pe) - 1
                    r = max(0, math.ceil(v))
                except Exception:
                    r = 0
            tbfr_t.append(T_SIGN_MS + T_TIP_MS + M * 1.0 + r * T_RTX_MS)
            arq_t.append(T_SIGN_MS + T_TIP_MS + M * 1.0
                         + (M-1) * (pe / max(1-pe, 1e-9)) * T_RTX_MS)
            rt = 1 - max(0, (M-1) * pe**(
                max(0, int((T_MAX_MS - T_SIGN_MS - T_TIP_MS - M*1.0) / T_RTX_MS))+1))
            rates.append(max(0.0, min(1.0, rt)) * 100)

        ax1.plot(PE_RANGE*100, tbfr_t, color=color, linewidth=1.8,
                 label=f"TBFR $M={M}$")
        ax1.plot(PE_RANGE*100, arq_t, color=color, linewidth=1.2,
                 linestyle="--", label=f"ARQ $M={M}$")
        ax2.plot(PE_RANGE*100, rates, color=color, linewidth=1.8,
                 label=f"TBFR $M={M}$")

    ax1.axhline(T_MAX_MS, color="#E63946", linestyle="-.", linewidth=1.5,
                label=f"$T_{{\\max}}={T_MAX_MS}$~ms")
    ax1.set(xlabel="$p_e$ (\\%)", ylabel="Latency (ms)",
            title="(D2a) Recovery Time: TBFR vs. ARQ",
            xlim=(0,25), ylim=(0, T_MAX_MS*2.5))
    ax1.legend(fontsize=8.5, ncol=2)

    ax2.axhline(GAMMA_REQ*100, color="#E63946", linestyle="-.", linewidth=1.5,
                label=f"$\\gamma_{{req}}=99.999\\%$")
    ax2.set(xlabel="$p_e$ (\\%)", ylabel="Success Rate (\\%)",
            title="(D2b) TBFR Success Rate",
            xlim=(0,25), ylim=(99.0, 100.01))
    ax2.legend(fontsize=9)

    plt.tight_layout()
    save(fig, "fig_D2_tbfr_vs_arq")


# ══════════════════════════════════════════════════════════════
#  FIGURE D3 — Attack Success Probability vs M
# ══════════════════════════════════════════════════════════════

def plot_D3_attack_vs_m():
    print("  Plotting fig_D3: Attack probability vs M...")
    analytical = [(M - 1) * 2**(-128) for M in M_RANGE]

    fig, ax = plt.subplots(figsize=(3.5, 4.5))
    ax.semilogy(M_RANGE, analytical, color="#E63946",
                linewidth=2.5, marker="o", markersize=6,
                label="Analytical bound (Theorem~1)")
    ax.axhline(2**(-128), color="#888", linestyle="--", linewidth=1.0,
               label="Quantum bound $2^{-128}$")
    ax.set(xlabel="Window Size $M$",
           ylabel="Attack Success Probability",
           title="(D3) Security vs. $M$",
           xlim=(1, max(M_RANGE)))
    ax.legend(fontsize=9)
    plt.tight_layout()
    save(fig, "fig_D3_attack_vs_m")


def main():
    print()
    print("=" * 60)
    print("  PQ-TDAG — Group D: Security Figures + W6 fix")
    print("=" * 60 + "\n")

    plot_D1_byzantine(n_gateways=9)
    plot_D4_byzantine_vs_n()
    plot_D2_tbfr()
    plot_D3_attack_vs_m()

    print()
    print("=" * 60)
    print("  DONE — Group D figures saved.")
    print("=" * 60)


if __name__ == "__main__":
    main()
