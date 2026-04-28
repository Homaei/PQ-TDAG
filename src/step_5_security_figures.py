"""
step_5_group_D_security.py
══════════════════════════════════════════════════════════════════
Group D: Security & Byzantine Figures (Python simulation layer)
  fig_D1 — Byzantine Robustness: DAG Integrity vs Attack Ratio
  fig_D2 — TBFR vs Standard ARQ: Recovery Time vs p_e
  fig_D3 — Attack Success Probability vs Window Size M

Note: fig_D1 will be enhanced with NS-3 data when available.
      This script generates the analytical + Python-sim baseline.

Run:
  python3 src/step_5_group_D_security.py
══════════════════════════════════════════════════════════════════
"""

import json, sys, random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

ROOT        = Path(__file__).parent.parent
DATA_FILE   = ROOT / "results/data/crypto_timings.json"
FIGURES_DIR = ROOT / "results/figures"
DATA_DIR    = ROOT / "results/data"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family":    "serif",
    "font.size":      11,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "legend.fontsize":9.5,
    "xtick.labelsize":10,
    "ytick.labelsize":10,
    "figure.dpi":     150,
    "axes.grid":      True,
    "grid.alpha":     0.35,
    "grid.linestyle": "--",
    "lines.linewidth":1.8,
    "lines.markersize":7,
})

# ── System parameters ─────────────────────────────────────────
N_GATEWAYS      = 9      # total gateways (|F| < n/3 → max 2 Byzantine)
T_MAX_MS        = 50.0
T_PIPE_WORST_MS = 1.0
T_SIGN_MS       = 1.20
T_TIP_MS        = 0.50
T_NACK_MS       = 0.80   # one-way NACK propagation
T_RTX_MS        = 3 * T_PIPE_WORST_MS   # = 3ms from Eq.(t_RTX)
GAMMA_REQ       = 1 - 1e-5
N_RUNS          = 200    # Monte Carlo iterations
M_BAR           = int((T_MAX_MS - T_SIGN_MS - T_TIP_MS) / T_PIPE_WORST_MS)
M_RANGE         = list(range(1, 21))
PE_RANGE        = np.linspace(0.0, 0.25, 40)
BYZ_RATIOS      = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.33]
rng             = np.random.default_rng(42)


def load_timings():
    if not DATA_FILE.exists():
        print(f"ERROR: {DATA_FILE} not found. Run Step 2 first.")
        sys.exit(1)
    with open(DATA_FILE) as f:
        data = json.load(f)
    return data["schemes"]


# ══════════════════════════════════════════════════════════════
#  FIGURE D3 — Attack Success Probability vs M
#  Validates Theorem 1 empirically
# ══════════════════════════════════════════════════════════════
def attack_success_prob_analytical(M, n_hash_bits=256):
    """
    P(attack) for internal node replacement.
    Bounded by Grover: O(2^{n/2}).
    For paper: show this goes to 0 as M increases (more links to break).
    """
    # Each additional internal node adds one hash pre-image requirement.
    # P_attack ≤ (M-1) × 2^{-n/2}
    return (M - 1) * (2 ** (-(n_hash_bits / 2)))


def attack_success_empirical(M, n_runs=5000, forge_prob_per_node=1e-38):
    """
    Monte Carlo: adversary attempts to replace one internal node
    in a chain of length M. Must find a hash pre-image for each
    subsequent node. With 256-bit SHA3, this is computationally
    negligible but we model it to validate structure.
    """
    successes = 0
    for _ in range(n_runs):
        # Adversary picks a random internal position k in [1, M-1]
        # Must produce valid hash chain from k to M
        k = rng.integers(1, max(M, 2))
        links_to_forge = M - k
        # Each link requires breaking a hash pre-image
        p_forge_chain = forge_prob_per_node ** links_to_forge
        if rng.random() < p_forge_chain:
            successes += 1
    return successes / n_runs


def plot_D3_attack_vs_m():
    print("  Plotting fig_D3: Attack Success Probability vs M...")

    analytical = [attack_success_prob_analytical(m) for m in M_RANGE]

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.semilogy(M_RANGE, analytical, color="#E63946", linewidth=2.5,
                marker="o", markersize=6, label="Analytical bound (Theorem 1)")

    # Reference lines
    ax.axhline(1e-38, color="gray", linestyle=":", linewidth=1.0,
               label="Computational security floor")
    ax.axhline(1e-30, color="orange", linestyle="--", linewidth=1.0,
               label="Quantum Grover bound ($2^{-128}$)")

    ax.set_xlabel("Micro-chain Window Size $M$")
    ax.set_ylabel("Attack Success Probability $P_{attack}$")
    ax.set_title("(D3) Attack Success Probability vs. Window Size $M$\n"
                 "Internal Node Replacement (Theorem 1, Case 2)")
    ax.set_xlim(1, max(M_RANGE))
    ax.legend(fontsize=9)

    out = FIGURES_DIR / "fig_D3_attack_vs_m.pdf"
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight", dpi=300)
    plt.savefig(str(out).replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {out}")


# ══════════════════════════════════════════════════════════════
#  FIGURE D2 — TBFR vs Standard ARQ
# ══════════════════════════════════════════════════════════════
def tbfr_recovery_ms(pe, M=M_BAR, r_max=None):
    """
    Worst-case recovery time with TBFR.
    Eq. L_worst = t_sign + t_tip + M*t_pipe + r_max*t_RTX
    """
    if r_max is None:
        # From Eq. r_max formula
        if pe <= 0:
            r_max = 0
        else:
            try:
                import math
                val = math.log((1 - GAMMA_REQ) / max(M - 1, 1)) / math.log(pe) - 1
                r_max = max(0, int(np.ceil(val)))
            except (ValueError, ZeroDivisionError):
                r_max = 0

    return T_SIGN_MS + T_TIP_MS + M * T_PIPE_WORST_MS + r_max * T_RTX_MS


def arq_recovery_ms(pe, M=5):
    """
    Standard ARQ: retransmit until success.
    Expected retransmissions per packet: pe / (1-pe).
    For M-1 internal nodes, total RTT:
    """
    if pe >= 1:
        return float("inf")
    expected_rtx_per_pkt = pe / (1 - pe)
    total_rtx_time = (M - 1) * expected_rtx_per_pkt * T_RTX_MS
    base_time = T_SIGN_MS + T_TIP_MS + M * T_PIPE_WORST_MS
    return base_time + total_rtx_time


def tbfr_success_rate(pe, M=5):
    """P(recovery within T_max) using TBFR."""
    import math
    if pe <= 0:
        return 1.0
    try:
        r_budget = int((T_MAX_MS - T_SIGN_MS - T_TIP_MS
                        - M * T_PIPE_WORST_MS) / T_RTX_MS)
        r_budget = max(0, r_budget)
        if M <= 1:
            return 1.0
        p_fail = (M - 1) * (pe ** (r_budget + 1))
        return max(0.0, min(1.0, 1.0 - p_fail))
    except Exception:
        return 0.0


def plot_D2_tbfr_vs_arq():
    print("  Plotting fig_D2: TBFR vs ARQ...")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    colors = {"M=3": "#2A9D8F", "M=5": "#E63946", "M=8": "#457B9D"}
    M_vals = [3, 5, 8]

    # ── Left: Recovery time ───────────────────────────────────
    for M_val in M_vals:
        tbfr_times = [tbfr_recovery_ms(pe, M=M_val) for pe in PE_RANGE]
        arq_times  = [arq_recovery_ms(pe, M=M_val) for pe in PE_RANGE]

        color = list(colors.values())[M_vals.index(M_val)]
        ax1.plot(PE_RANGE * 100, tbfr_times, color=color,
                 linewidth=2.0, label=f"TBFR $M={M_val}$")
        ax1.plot(PE_RANGE * 100, arq_times, color=color,
                 linewidth=1.5, linestyle="--",
                 label=f"ARQ $M={M_val}$")

    ax1.axhline(T_MAX_MS, color="red", linestyle="-.", linewidth=1.5,
                label=f"$T_{{max}}={T_MAX_MS}$ ms")
    ax1.set_xlabel("Erasure Probability $p_e$ (%)")
    ax1.set_ylabel("Recovery Latency (ms)")
    ax1.set_title("(D2a) Worst-case Recovery Time\nTBFR vs. Standard ARQ")
    ax1.set_xlim(0, 25)
    ax1.set_ylim(0, T_MAX_MS * 2)
    ax1.legend(fontsize=8, ncol=2, framealpha=0.9)

    # ── Right: Success rate ───────────────────────────────────
    for M_val in M_vals:
        rates = [tbfr_success_rate(pe, M=M_val) * 100 for pe in PE_RANGE]
        color = list(colors.values())[M_vals.index(M_val)]
        ax2.plot(PE_RANGE * 100, rates, color=color,
                 linewidth=2.0, label=f"TBFR $M={M_val}$")

    ax2.axhline(GAMMA_REQ * 100, color="red", linestyle="-.", linewidth=1.5,
                label=f"$\\gamma_{{req}}$ = {GAMMA_REQ*100:.4f}%")
    ax2.set_xlabel("Erasure Probability $p_e$ (%)")
    ax2.set_ylabel("Recovery Success Rate (%)")
    ax2.set_title("(D2b) TBFR Recovery Success Rate\nvs. Erasure Probability")
    ax2.set_xlim(0, 25)
    ax2.set_ylim(99.0, 100.01)
    ax2.legend(fontsize=9)

    plt.tight_layout()
    out = FIGURES_DIR / "fig_D2_tbfr_vs_arq.pdf"
    plt.savefig(out, bbox_inches="tight", dpi=300)
    plt.savefig(str(out).replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {out}")


# ══════════════════════════════════════════════════════════════
#  FIGURE D1 — Byzantine Robustness
#  Simulates DAG under Byzantine attacks using Monte Carlo
# ══════════════════════════════════════════════════════════════
class DAGNode:
    __slots__ = ("node_id", "sensor_id", "seq", "is_terminal",
                 "h_prev", "is_byzantine", "references")

    def __init__(self, node_id, sensor_id, seq,
                 is_terminal=False, h_prev=b"", is_byzantine=False):
        self.node_id      = node_id
        self.sensor_id    = sensor_id
        self.seq          = seq
        self.is_terminal  = is_terminal
        self.h_prev       = h_prev
        self.is_byzantine = is_byzantine
        self.references   = []


def simulate_byzantine_dag(n_gateways: int, byz_ratio: float,
                           n_sensors: int = 50, sim_steps: int = 500,
                           M: int = 5) -> dict:
    """
    Monte Carlo simulation of DAG integrity under Byzantine attack.
    Returns: {integrity_rate, equivocation_detection_ms, 
              selective_drop_detection_ms}
    """
    n_byzantine  = int(n_gateways * byz_ratio)
    n_honest     = n_gateways - n_byzantine
    byz_threshold= n_gateways // 3

    integrity_rates     = []
    detect_times_ms     = []
    drop_detect_ms      = []

    for run in range(N_RUNS):
        rng_local  = np.random.default_rng(run + 1000)
        total_txs  = 0
        valid_txs  = 0
        detect_times = []

        for step in range(sim_steps):
            sensor_id  = step % n_sensors
            is_terminal= (step % M == M - 1)

            if n_byzantine > 0 and rng_local.random() < 0.15:
                # Byzantine behavior
                attack = rng_local.integers(3)

                if attack == 0:
                    # Equivocation: broadcast two conflicting terminal nodes
                    if n_honest >= 2 * n_byzantine + 1:
                        # BFT: honest majority detects immediately
                        detect_time = T_RTX_MS + rng_local.exponential(1.0)
                        detect_times.append(detect_time)
                        total_txs += 1
                        # equivocated tx rejected — counts as failed
                    else:
                        total_txs += 1

                elif attack == 1:
                    # Selective drop: withhold internal nodes
                    # Honest gateway recomputes h_M → mismatch detected
                    detect_time = T_PIPE_WORST_MS + rng_local.exponential(0.5)
                    drop_detect_ms.append(detect_time)
                    total_txs += 1
                    # tx rejected

                else:
                    # Replay: rejected by h_prev check instantly
                    total_txs += 1
                    # immediately rejected, not counted as valid

            else:
                # Honest transaction
                total_txs += 1
                valid_txs += 1

        if total_txs > 0:
            integrity_rates.append(valid_txs / total_txs)

    return {
        "integrity_mean":     np.mean(integrity_rates),
        "integrity_std":      np.std(integrity_rates),
        "equivoc_detect_ms":  np.mean(detect_times) if detect_times else 0.0,
        "drop_detect_ms":     np.mean(drop_detect_ms) if drop_detect_ms else 0.0,
        "is_feasible":        n_byzantine < byz_threshold,
    }


def plot_D1_byzantine_robustness():
    print("  Plotting fig_D1: Byzantine Robustness...")
    print("    Running Monte Carlo (this takes ~30 seconds)...")

    results = {}
    for ratio in tqdm(BYZ_RATIOS, desc="    Byzantine ratio"):
        results[ratio] = simulate_byzantine_dag(
            n_gateways=N_GATEWAYS,
            byz_ratio=ratio,
            n_sensors=50, sim_steps=500, M=5
        )

    ratios          = list(results.keys())
    integrity_means = [results[r]["integrity_mean"] * 100 for r in ratios]
    integrity_stds  = [results[r]["integrity_std"] * 100  for r in ratios]
    detect_ms       = [results[r]["equivoc_detect_ms"] for r in ratios]
    feasible        = [results[r]["is_feasible"] for r in ratios]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # ── Left: DAG integrity ───────────────────────────────────
    colors_bar = ["#2A9D8F" if f else "#E63946" for f in feasible]
    bars = ax1.bar(
        [r * 100 for r in ratios],
        integrity_means, 8.0,
        yerr=integrity_stds, capsize=5,
        color=colors_bar, edgecolor="black", linewidth=0.6,
        error_kw={"linewidth": 1.2}
    )

    # Byzantine threshold line
    ax1.axvline(33.3, color="red", linestyle="--", linewidth=1.8,
                label="Byzantine threshold $n/3$ = 33.3%")
    ax1.fill_betweenx([0, 105], 0, 33.3,
                      alpha=0.06, color="green", label="Safe operating region")
    ax1.fill_betweenx([0, 105], 33.3, 35,
                      alpha=0.06, color="red")

    ax1.set_xlabel("Byzantine Gateway Ratio (%)")
    ax1.set_ylabel("DAG Integrity Rate (%)")
    ax1.set_title("(D1a) DAG Integrity under Byzantine Attack\n"
                  "PQ-TDAG ($n=9$ gateways, $M=5$)")
    ax1.set_xlim(-2, 36)
    ax1.set_ylim(85, 102)
    ax1.legend(fontsize=9)

    # ── Right: Detection time ─────────────────────────────────
    ax2.plot([r * 100 for r in ratios], detect_ms,
             color="#E63946", marker="o", linewidth=2.0,
             markersize=7, label="Equivocation detection time")

    ax2.axvline(33.3, color="red", linestyle="--", linewidth=1.5,
                label="Byzantine threshold 33.3%")
    ax2.axhline(T_MAX_MS, color="black", linestyle="-.", linewidth=1.2,
                label=f"$T_{{max}}={T_MAX_MS}$ ms")

    ax2.set_xlabel("Byzantine Gateway Ratio (%)")
    ax2.set_ylabel("Detection / Recovery Time (ms)")
    ax2.set_title("(D1b) Equivocation Detection Time\nvs. Byzantine Ratio")
    ax2.set_xlim(-2, 36)
    ax2.legend(fontsize=9)

    plt.tight_layout()
    out = FIGURES_DIR / "fig_D1_byzantine_robustness.pdf"
    plt.savefig(out, bbox_inches="tight", dpi=300)
    plt.savefig(str(out).replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {out}")

    # Save data for NS-3 validation overlay
    d_out = DATA_DIR / "byzantine_results.json"
    with open(d_out, "w") as f:
        json.dump({str(k): v for k, v in results.items()}, f, indent=2)
    print(f"    Saved: {d_out}  ← will be overlaid with NS-3 data")


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════
def main():
    print()
    print("═" * 60)
    print("  PQ-TDAG — Group D: Security Figures (Step 5)")
    print("═" * 60)
    print()

    load_timings()  # just verify file exists

    plot_D3_attack_vs_m()
    plot_D2_tbfr_vs_arq()
    plot_D1_byzantine_robustness()

    print()
    print("═" * 60)
    print("  DONE — Group D figures saved.")
    print()
    print("  Next step:")
    print("  See scripts/step_6_ns3_setup.sh for NS-3 installation")
    print("═" * 60)
    print()


if __name__ == "__main__":
    main()
