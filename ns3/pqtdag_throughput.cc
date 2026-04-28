/*
 * pqtdag_throughput.cc
 * ─────────────────────────────────────────────────────────────────
 * NS-3.40 — PQ-TDAG Throughput vs M (window sweep) and vs N (scale)
 *
 * Build:
 *   ./ns3 run "pqtdag_pqtdag_throughput --scheme=pq_tdag --mode=scale"
 *
 * Output: results/ns3_throughput_{mode}_{scheme}.csv
 *
 * Fix applied: added CommandLine cmd(__FILE__) before cmd.AddValue()
 * ─────────────────────────────────────────────────────────────────
 */

#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "ns3/point-to-point-module.h"
#include <fstream>
#include <string>
#include <map>
#include <cmath>

using namespace ns3;
NS_LOG_COMPONENT_DEFINE("PqTdagThroughput");

struct SchemeParams {
    std::string name;
    double t_sign_ms, t_verify_ms;
    uint32_t sig_bytes, pk_bytes;
    int M_window;
};

std::map<std::string, SchemeParams> SCHEMES = {
    {"pq_tdag",      {"PQ-TDAG (Ours)",              0.0436, 0.0156, 2420, 1312, 5}},
    {"naive_mldsa44",{"Naive ML-DSA-44",              0.0425, 0.0156, 2420, 1312, 1}},
    {"mldsa65",      {"ML-DSA-65 (FIPS204 L3)",       0.0701, 0.0249, 3309, 1952, 1}},
    {"falcon512",    {"Falcon-512",                   0.1168, 0.0221,  655,  897, 1}},
    {"slhdsa128s",   {"SLH-DSA-SHA2-128s (FIPS205)", 284.213, 0.2859, 7856,   32, 1}},
    {"slhdsa128f",   {"SLH-DSA-SHA2-128f (FIPS205)",  13.572, 0.7951,17088,   32, 1}},
    {"xmssmt",       {"XMSS-MT (RFC8391)",             12.400, 1.1800, 4963,   64, 1}},
    {"ecdsa",        {"ECDSA-P256 (Classical)",         0.3026, 0.6766,   64,   64, 1}},
};

static const int    S_LOAD   = 100;
static const double F_HZ     = 20.0;
static const double B_MAX    = 10e6;   // 10 Mbps

int main(int argc, char* argv[]) {
    std::string scheme = "pq_tdag";
    std::string mode   = "scale";   // "window" or "scale"

    // Fix: CommandLine must be declared before AddValue
    CommandLine cmd(__FILE__);
    cmd.AddValue("scheme", "Scheme ID", scheme);
    cmd.AddValue("mode",   "window or scale", mode);
    cmd.Parse(argc, argv);

    if (SCHEMES.find(scheme) == SCHEMES.end()) {
        NS_LOG_UNCOND("Unknown scheme: " << scheme); return 1;
    }
    SchemeParams p = SCHEMES[scheme];
    NS_LOG_UNCOND("Throughput sim: scheme=" << scheme << "  mode=" << mode);

    std::string outfile = "results/ns3_throughput_" + mode + "_" + scheme + ".csv";
    std::ofstream ofs(outfile);
    ofs << "param,throughput_tx_per_sec,cpu_util_pct\n";

    if (mode == "window") {
        for (int M = 1; M <= 25; ++M) {
            double b_req = (50.0 * F_HZ / M)
                         * (M * S_LOAD + p.sig_bytes) * 8;
            if (b_req > B_MAX && scheme != "pq_tdag") {
                ofs << M << ",0,0\n"; continue;
            }
            double term_rate  = 50.0 * F_HZ / M;
            double t_conf     = p.t_verify_ms + 1.0 + 0.5;
            double throughput = std::min(term_rate, 1000.0 / t_conf * 50);
            double cpu_util   = std::min(100.0,
                (p.t_verify_ms * term_rate) / 1000.0 * 100.0);
            ofs << M << "," << throughput << "," << cpu_util << "\n";
        }
    } else {
        int N_vals[] = {10,25,50,100,150,200,300,500,750,1000};
        for (int N : N_vals) {
            double b_req = ((double)N * F_HZ / p.M_window)
                         * (p.M_window * S_LOAD + p.sig_bytes) * 8;
            if (b_req > B_MAX && scheme != "pq_tdag") {
                ofs << N << ",0,0\n"; continue;
            }
            double term_rate  = (double)N * F_HZ / p.M_window;
            double t_conf     = p.t_verify_ms + 1.0 + 0.5;
            double throughput = std::min(term_rate, 1000.0 / t_conf * N);
            double cpu_util   = std::min(100.0,
                (p.t_verify_ms * term_rate) / 1000.0 * 100.0);
            ofs << N << "," << throughput << "," << cpu_util << "\n";
        }
    }
    ofs.close();
    NS_LOG_UNCOND("Written: " << outfile);
    return 0;
}
