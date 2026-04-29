"""
src/step_sensitivity_tpipe.py
───────────────────────────────────────────────────────────────────
Sensitivity Analysis for Network Pipe Delay (t_pipe)

The baseline simulation model uses a fixed t_pipe = 1 ms (ideal 5G URLLC link).
In practical industrial deployments, physical factors such as multipath fading, 
RF interference from heavy machinery, and MAC-layer scheduling can increase 
the propagation delay to 3–10 ms.

This script performs a sensitivity sweep across:
  t_pipe ∈ {1, 3, 5, 10} ms

For each t_pipe value, the analysis computes:
  1. L_nominal (baseline confirmation latency) vs M
  2. The maximum feasible window size (M_max) ensuring real-time bounds
  3. Safety margin evaluation confirming stable architectural behavior under fading

Output:
  results/figures/fig_sensitivity_tpipe.pdf
  results/logs/sensitivity_tpipe_report.txt
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ROOT        = Path(__file__).parent.parent
DATA_FILE   = ROOT / "results/data/crypto_timings.json"
FIGURES_DIR = ROOT / "results/figures"; FIGURES_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR    = ROOT / "results/logs";    LOGS_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "font.size": 11,
    "axes.labelsize": 12, "axes.titlesize": 11,
    "legend.fontsize": 9, "xtick.labelsize": 10,
    "ytick.labelsize": 10, "figure.dpi": 150,
    "axes.grid": True, "grid.alpha": 0.35, "grid.linestyle": "--",
})

# ── System parameters ─────────────────────────────────────────────
T_MAX_MS  = 50.0
T_RTX_MS  = 3.0     # retransmission round-trip time (NACK + retx)
T_TIP_MS  = 0.5     # tip selection latency
GAMMA_REQ = 1 - 1e-5
M_RANGE   = np.arange(1, 16)

# t_pipe sweep values (ms)
# 1 ms = ideal 5G URLLC
# 3 ms = moderate fading / MAC backpressure
# 5 ms = heavy RF interference
# 10 ms = worst-case ICS channel
T_PIPE_SWEEP = [1.0, 3.0, 5.0, 10.0]
COLORS_SWEEP = ["#E63946", "#2A9D8F", "#457B9D", "#6A4C93"]
STYLES_SWEEP = ["-", "--", "-.", ":"]


def load_t_sign():
    """Load measured sign time from framework benchmarks."""
    if DATA_FILE.exists():
        with open(DATA_FILE) as f:
            d = json.load(f)
        return d["schemes"].get("pq_tdag", {}).get("t_sign_mean_ms", 0.0436)
    return 0.0436


def L_nominal(M, t_pipe, t_sign):
    """
    Baseline confirmation latency without channel errors.
    This calculates the system latency prior to TBFR recovery overhead.
    """
    return t_sign + T_TIP_MS + M * t_pipe


def r_max_formula(M, t_pipe, t_sign):
    """
    Maximum TBFR retransmissions allowed within the temporal deadline.
    """
    slack = T_MAX_MS - t_sign - T_TIP_MS - M * t_pipe
    if slack <= 0:
        return 0
    return max(0, int(slack / T_RTX_MS))


def safety_margin(M, t_pipe, t_sign):
    """
    Ratio evaluating system timing headroom before error mitigation.
    """
    ln = L_nominal(M, t_pipe, t_sign)
    return T_MAX_MS / ln if ln > 0 else 0.0


def plot_sensitivity(t_sign_ms: float):
    """
    Generates the sensitivity figure.
    """
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))
    fig.suptitle(
        "Sensitivity Analysis: Network Pipe Delay $t_{\\mathrm{pipe}}$\n"
        "PQ-TDAG Framework, "
        f"$t_{{\\mathrm{{sign}}}}={t_sign_ms:.3f}$~ms, "
        f"$T_{{\\mathrm{{max}}}}={T_MAX_MS}$~ms",
        fontsize=10, y=1.02
    )

    ax1, ax2, ax3 = axes
    lines_summary = []

    for t_pipe, color, ls in zip(T_PIPE_SWEEP, COLORS_SWEEP, STYLES_SWEEP):
        l_vals  = []
        sm_vals = []
        rb_vals = []

        for M in M_RANGE:
            r   = r_max_formula(M, t_pipe, t_sign_ms)
            ln  = L_nominal(M, t_pipe, t_sign_ms)
            sm  = safety_margin(M, t_pipe, t_sign_ms)
            l_vals.append(ln)
            sm_vals.append(sm)
            rb_vals.append(r)

        label = f"$t_{{pipe}}={t_pipe:.0f}$~ms"

        ax1.plot(M_RANGE, l_vals, color=color, linestyle=ls,
                 linewidth=2.0, marker="o", markersize=4, label=label)
        ax2.plot(M_RANGE, sm_vals, color=color, linestyle=ls,
                 linewidth=2.0, marker="o", markersize=4, label=label)
        ax3.plot(M_RANGE, rb_vals, color=color, linestyle=ls,
                 linewidth=2.0, marker="o", markersize=4, label=label)

        feasible_M = [M for M in M_RANGE if r_max_formula(M, t_pipe, t_sign_ms) >= 1]
        m_max = max(feasible_M) if feasible_M else None
        lines_summary.append((t_pipe, m_max))

    # Panel (a): Nominal latency
    ax1.axhline(T_MAX_MS, color="red", linestyle="-.", linewidth=1.5,
                label=f"$T_{{max}}={T_MAX_MS}$~ms")
    ax1.set(xlabel="Window Size $M$",
            ylabel="$L_{\\mathrm{nominal}}$ (ms)",
            title="(a) Nominal Latency vs. $M$",
            xlim=(1, M_RANGE[-1]), ylim=(0, T_MAX_MS * 1.6))
    ax1.legend(fontsize=8.5)

    # Panel (b): Safety margin evaluation
    ax2.axhline(1.0, color="red", linestyle="-.", linewidth=1.5,
                label="Deadline boundary")
    ax2.axhline(16.0, color="green", linestyle=":", linewidth=1.0,
                alpha=0.7, label="16× margin baseline")
    ax2.fill_between(M_RANGE, 1.0, 16.0, alpha=0.05, color="green")
    ax2.set(xlabel="Window Size $M$",
            ylabel="$T_{\\mathrm{max}} / L_{\\mathrm{nominal}}$",
            title="(b) Safety Margin vs. $M$",
            xlim=(1, M_RANGE[-1]), ylim=(0, 25))
    ax2.legend(fontsize=8.5)

    # Panel (c): Retransmission budget
    ax3.axhline(0, color="red", linestyle="-.", linewidth=1.0,
                label="$r_{max}=0$ (no recovery bounds)")
    ax3.set(xlabel="Window Size $M$",
            ylabel="Retransmission Budget $r_{\\mathrm{max}}$",
            title="(c) TBFR Budget vs. $M$",
            xlim=(1, M_RANGE[-1]))
    ax3.legend(fontsize=8.5)

    plt.tight_layout()
    out = FIGURES_DIR / "fig_sensitivity_tpipe.pdf"
    plt.savefig(out, bbox_inches="tight", dpi=300)
    plt.savefig(str(out).replace(".pdf",".png"), bbox_inches="tight", dpi=150)
    plt.close()
    print(f"    Saved: {out}")

    return lines_summary


def generate_report(t_sign_ms: float, summary: list) -> str:
    """
    Generates summary report for the sensitivity analysis.
    """
    lines = [
        "",
        "=" * 65,
        "  SENSITIVITY ANALYSIS REPORT",
        "=" * 65,
        "",
        "── EXECUTION PARAMETERS ────────────────────────────────────",
        "",
        f"  t_sign (measured) = {t_sign_ms:.4f} ms",
        f"  T_max             = {T_MAX_MS} ms",
        f"  T_RTX             = {T_RTX_MS} ms",
        "",
        f"  {'t_pipe (ms)':>12}  {'Max feasible M':>15}  {'Safety @ M=5':>14}",
        "  " + "-"*45,
    ]
    for t_pipe, m_max in summary:
        sm = safety_margin(5, t_pipe, t_sign_ms)
        m_str = str(m_max) if m_max else "infeasible"
        lines.append(f"  {t_pipe:>12.1f}  {m_str:>15}  {sm:>12.1f}×")

    lines += [
        "",
        "=" * 65,
    ]
    return "\n".join(lines)


def main():
    print()
    print("=" * 60)
    print("  PQ-TDAG — Sensitivity Analysis: t_pipe")
    print("=" * 60 + "\n")

    t_sign_ms = load_t_sign()
    print(f"  t_sign (measured) = {t_sign_ms:.4f} ms")
    print(f"  T_max = {T_MAX_MS} ms")
    print(f"  t_pipe sweep parameters: {T_PIPE_SWEEP} ms\n")

    summary = plot_sensitivity(t_sign_ms)

    print()
    print(f"  {'t_pipe (ms)':>12}  {'Max feasible M':>15}  {'Safety @ M=5':>14}")
    print("  " + "-"*45)
    for t_pipe, m_max in summary:
        sm    = safety_margin(5, t_pipe, t_sign_ms)
        m_str = str(m_max) if m_max else "infeasible"
        print(f"  {t_pipe:>12.1f}  {m_str:>15}  {sm:>12.1f}×")

    report = generate_report(t_sign_ms, summary)
    out    = LOGS_DIR / "sensitivity_tpipe_report.txt"
    out.write_text(report)
    print(f"\n  Saved: {out}")

    print()
    print("=" * 60)
    print("  DONE — Sensitivity evaluation completed.")
    print("=" * 60)


if __name__ == "__main__":
    main()
