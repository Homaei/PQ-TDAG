"""
step_10_final_tables.py
Generates LaTeX tables for the paper from measured data.
"""
import json
import numpy as np
from pathlib import Path

ROOT       = Path(__file__).parent.parent
DATA_FILE  = ROOT / "results/data/crypto_timings.json"
TABLES_DIR = ROOT / "results/tables"
TABLES_DIR.mkdir(parents=True, exist_ok=True)

B_MAX_MBPS = 10.0
S_PAYLOAD  = 100
N_REF, F_REF, M_REF = 50, 20, 5


def load_timings():
    with open(DATA_FILE) as f:
        return json.load(f)


def bw_mbps(N, f, sig_bytes, M=1):
    return N * (f / M) * (M * S_PAYLOAD + sig_bytes) * 8 / 1e6


def ics_feasible_cell(sid, sig_bytes):
    M  = M_REF if sid == "pq_tdag" else 1
    bw = bw_mbps(N_REF, F_REF, sig_bytes, M)
    if bw <= B_MAX_MBPS:
        return r"\cellcolor{green!10}\checkmark"
    elif bw <= 15.0:
        return r"\cellcolor{orange!20}$\approx$"
    else:
        return r"\cellcolor{red!10}\ding{55}"


SCHEME_ORDER = [
    "pq_tdag", "naive_mldsa44", "mldsa65", "falcon512",
    "slhdsa128s", "slhdsa128f", "xmssmt", "ecdsa"
]


def gen_table1(data: dict) -> str:
    schemes = data["schemes"]
    meta    = data["metadata"]
    liboqs_ver = meta.get("liboqs_version", "0.15.0")
    # --- FIXED: correct CPU string ---
    cpu_str = "Intel Core i9-14900KF with AVX2 and AVX-VNNI"

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Cryptographic Baseline Comparison. "
        r"All gateway timings measured on " + cpu_str +
        r" via liboqs~v" + liboqs_ver +
        r". Bold = proposed method.}",
        r"\label{tab:baselines_full}",
        r"\begin{threeparttable}",
        r"\begin{tabular}{lrrrrrrcc}",
        r"\toprule",
        r"\textbf{Scheme} & $|\sigma|$ (B) & $|pk|$ (B) & "
        r"$t_{\mathrm{sign}}$ (ms) & $t_{\mathrm{ver}}$ (ms) & "
        r"$B_{\mathrm{req}}^{N=50}$ (Mbps) & Sign/s & PQC & ICS \\",
        r"\midrule",
    ]

    for sid in SCHEME_ORDER:
        if sid not in schemes:
            continue
        s = schemes[sid]
        if "error" in s or "t_sign_mean_ms" not in s:
            continue

        M    = M_REF if sid == "pq_tdag" else 1
        bw   = bw_mbps(N_REF, F_REF, s.get("sig_bytes", 0), M)
        pqc  = r"\checkmark" if s.get("is_pqc", False) else r"$\times$"
        feas = ics_feasible_cell(sid, s.get("sig_bytes", 0))
        sps  = int(s.get("sign_ops_per_sec", 0))
        is_ref = s.get("source", "").startswith("NIST") or \
                 s.get("n_iterations") == "reference"

        label = s.get("label", sid)
        if is_ref:
            label += r"~$[\dagger]$"

        row = [
            r"\textbf{" + label + r"}" if sid == "pq_tdag" else label,
            f"{s.get('sig_bytes', 0)}",
            f"{s.get('pk_bytes', 0)}",
            f"${s.get('t_sign_mean_ms', 0):.3f} \\pm {s.get('t_sign_std_ms', 0):.3f}$",
            f"${s.get('t_verify_mean_ms', 0):.3f} \\pm {s.get('t_verify_std_ms', 0):.3f}$",
            f"{bw:.2f}",
            f"{sps:,}",
            pqc,
            feas,
        ]
        lines.append(" & ".join(row) + r" \\")
        if sid == "pq_tdag":
            lines.append(r"\midrule")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\begin{tablenotes}\small",
        r"\item $|\sigma|$: signature size; $|pk|$: public key size; "
        r"$B_{\mathrm{req}}$: required bandwidth at $N=50$, $f=20$~Hz; "
        r"$M=" + str(M_REF) + r"$ for PQ-TDAG, $M=1$ for all others.",
        r"\item $[\dagger]$ Literature reference value (RFC~8391, pqm4 benchmarks).",
        r"\item \cellcolor{red!10} = infeasible; \cellcolor{green!10} = feasible.",
        r"\end{tablenotes}",
        r"\end{threeparttable}",
        r"\end{table*}",
    ]
    return "\n".join(lines)


def gen_table2(data: dict) -> str:
    schemes = data["schemes"]

    PROPERTIES = {
        "pq_tdag":      (True,  True,  True,  True,  "Decentralised DAG"),
        "naive_mldsa44":(True,  True,  True,  False, "Flat signed DAG"),
        "mldsa65":      (True,  True,  True,  False, "Flat signed DAG"),
        "falcon512":    (True,  True,  True,  False, "Flat signed DAG"),
        "slhdsa128s":   (True,  True,  True,  False, "Flat signed DAG"),
        "slhdsa128f":   (True,  True,  True,  False, "Flat signed DAG"),
        "xmssmt":       (True,  True,  False, False, "Stateful --- key reuse risk"),
        "ecdsa":        (False, False, True,  False, "Quantum-vulnerable"),
    }
    ck = r"\checkmark"
    xx = r"\ding{55}"

    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Security Properties Comparison.}",
        r"\label{tab:security}",
        r"\begin{tabular}{lccccp{2.6cm}}",
        r"\toprule",
        r"\textbf{Scheme} & EUF-CMA & PQ & Stateless & BFT DAG & Notes \\",
        r"\midrule",
    ]

    for sid in SCHEME_ORDER:
        if sid not in PROPERTIES:
            continue
        label = schemes.get(sid, {}).get("label", sid)
        euf, pq, sl, bft, note = PROPERTIES[sid]
        row = [
            r"\textbf{" + label + r"}" if sid == "pq_tdag" else label,
            ck if euf else xx,
            ck if pq  else xx,
            ck if sl  else r"\textbf{\ding{55}}",
            ck if bft else xx,
            note,
        ]
        lines.append(" & ".join(row) + r" \\")
        if sid == "pq_tdag":
            lines.append(r"\midrule")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def main():
    print("\n" + "="*55)
    print("  PQ-TDAG — Final Table Generator (Step 10)")
    print("="*55 + "\n")

    data = load_timings()
    print(f"  Loaded {len(data['schemes'])} schemes")

    t1 = gen_table1(data)
    t2 = gen_table2(data)

    out1 = TABLES_DIR / "table1_baselines.tex"
    out2 = TABLES_DIR / "table2_security.tex"
    out1.write_text(t1)
    out2.write_text(t2)

    print(f"  Saved: {out1}")
    print(f"  Saved: {out2}")

    print("\n  ALL STEPS COMPLETE.")
    print("="*55 + "\n")


if __name__ == "__main__":
    main()
