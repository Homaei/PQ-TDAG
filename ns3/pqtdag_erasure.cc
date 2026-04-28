/*
 * pqtdag_erasure.cc
 * ─────────────────────────────────────────────────────────────────
 * NS-3.40 — TBFR erasure resilience validation (Lemma 3)
 * Sweeps packet erasure probability p_e for multiple M values.
 *
 * Build:
 *   ./ns3 run "pqtdag_pqtdag_erasure --M=5"
 *
 * Output: results/ns3_erasure_M{M}.csv
 * ─────────────────────────────────────────────────────────────────
 */

#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "ns3/point-to-point-module.h"
#include <fstream>
#include <vector>
#include <cmath>

using namespace ns3;
NS_LOG_COMPONENT_DEFINE("PqTdagErasure");

// Injected from liboqs measurements (i9-14900KF)
static double T_SIGN_MS       = 0.0436;
static double T_VERIFY_MS     = 0.0156;
static double T_PIPE_WORST_MS = 1.0;
static double T_RTX_MS        = 3.0;
static double T_TIP_MS        = 0.5;
static double T_MAX_MS        = 50.0;

struct ChainResult {
    bool   success;
    double latency_ms;
};

ChainResult SimulateChain(int M, double pe, int r_budget,
                          Ptr<UniformRandomVariable> rng) {
    double total = T_SIGN_MS + T_TIP_MS;
    bool   ok    = true;

    for (int k = 0; k < M - 1; ++k) {
        bool delivered = false;
        for (int attempt = 0; attempt <= r_budget; ++attempt) {
            total += T_PIPE_WORST_MS;
            if (rng->GetValue() >= pe) {
                delivered = true; break;
            }
            if (attempt < r_budget) total += T_RTX_MS;
        }
        if (!delivered) { ok = false; break; }
    }
    total += T_PIPE_WORST_MS + T_VERIFY_MS;
    return {ok, total};
}

int main(int argc, char* argv[]) {
    int M = 5;
    CommandLine cmd(__FILE__);
    cmd.AddValue("M",        "Window size",          M);
    cmd.AddValue("t_sign",   "Sign latency (ms)",    T_SIGN_MS);
    cmd.AddValue("t_verify", "Verify latency (ms)",  T_VERIFY_MS);
    cmd.Parse(argc, argv);

    int r_budget = std::max(0, (int)std::floor(
        (T_MAX_MS - T_SIGN_MS - T_TIP_MS - M * T_PIPE_WORST_MS)
        / T_RTX_MS));

    NS_LOG_UNCOND("Erasure sim: M=" << M
        << "  r_budget=" << r_budget
        << "  t_sign=" << T_SIGN_MS << "ms");

    std::string fname = "results/ns3_erasure_M" + std::to_string(M) + ".csv";
    std::ofstream ofs(fname);
    ofs << "pe,latency_ms_mean,latency_ms_std,success_rate,deadline_ok\n";

    Ptr<UniformRandomVariable> rng = CreateObject<UniformRandomVariable>();
    const int N_RUNS   = 2000;
    const int PE_STEPS = 40;

    for (int step = 0; step <= PE_STEPS; ++step) {
        double pe = step * 0.25 / PE_STEPS;

        std::vector<double> lats;
        int successes = 0;
        for (int run = 0; run < N_RUNS; ++run) {
            auto res = SimulateChain(M, pe, r_budget, rng);
            lats.push_back(res.latency_ms);
            if (res.success && res.latency_ms <= T_MAX_MS) successes++;
        }

        double mean = 0, var = 0;
        for (double v : lats) mean += v;
        mean /= lats.size();
        for (double v : lats) var += (v - mean) * (v - mean);
        double stddev = std::sqrt(var / lats.size());
        double sr     = (double)successes / N_RUNS;

        ofs << pe << "," << mean << "," << stddev << ","
            << sr << "," << (mean <= T_MAX_MS ? 1 : 0) << "\n";
    }
    ofs.close();
    NS_LOG_UNCOND("Written: " << fname);
    return 0;
}
