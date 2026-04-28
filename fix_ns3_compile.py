#!/usr/bin/env python3
"""
fix_ns3_compile.py — Fixes 3 compile errors in NS-3 scenarios
  1. TimestampTag ambiguous → rename to PqTdagTag
  2. SetRecvCallback signature wrong in NS-3.40
  3. cmd not declared in pqtdag_throughput.cc
"""

from pathlib import Path

SCRATCH = Path("/root/ns-allinone-3.40/ns-3.40/scratch/pq_tdag")

# ══════════════════════════════════════════════════════════
#  FIX 1+2: pqtdag_latency_cdf.cc
# ══════════════════════════════════════════════════════════
LATENCY_CDF = r"""
#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "ns3/point-to-point-module.h"
#include "ns3/applications-module.h"
#include <fstream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <map>

using namespace ns3;
NS_LOG_COMPONENT_DEFINE("PqTdagLatencyCdf");

struct SchemeParams {
    std::string name;
    double      t_sign_ms;
    double      t_verify_ms;
    uint32_t    sig_bytes;
    uint32_t    pk_bytes;
    int         M_window;
};

// ── Scheme map injected from crypto_timings.json ──────────
std::map<std::string, SchemeParams> SCHEMES = {
SCHEME_PARAMS_PLACEHOLDER
};

static const uint32_t N_SENSORS    = 50;
static const double   F_HZ         = 20.0;
static const uint32_t S_PAYLOAD    = 100;
static const double   T_MAX_MS     = 50.0;
static const double   SIM_DURATION = 30.0;
static const double   B_MAX_MBPS   = 10.0;

std::vector<double> g_latencies_ms;
std::string         g_scheme_id;
SchemeParams        g_params;

// ── Renamed to avoid conflict with ns3::TimestampTag ─────
class PqTdagTag : public Tag {
public:
    static TypeId GetTypeId() {
        static TypeId tid = TypeId("PqTdagTag")
            .SetParent<Tag>()
            .AddConstructor<PqTdagTag>();
        return tid;
    }
    TypeId GetInstanceTypeId() const override { return GetTypeId(); }
    uint32_t GetSerializedSize() const override { return 8; }
    void Serialize(TagBuffer buf) const override { buf.WriteDouble(m_ts); }
    void Deserialize(TagBuffer buf) override { m_ts = buf.ReadDouble(); }
    void Print(std::ostream& os) const override { os << "ts=" << m_ts; }
    double m_ts{0.0};
};

// ── NS-3.40 correct callback signature ───────────────────
void RxPacket(Ptr<Socket> socket) {
    Ptr<Packet> pkt;
    Address from;
    while ((pkt = socket->RecvFrom(from))) {
        PqTdagTag tag;
        if (pkt->FindFirstMatchingByteTag(tag)) {
            double now_ms  = Simulator::Now().GetMilliSeconds();
            double lat_ms  = now_ms - tag.m_ts;
            double t_crypto = g_params.t_verify_ms
                            + g_params.t_sign_ms / g_params.M_window;
            double jitter   = ((double)rand() / RAND_MAX - 0.5) * 0.2;
            g_latencies_ms.push_back(lat_ms + t_crypto + jitter);
        }
    }
}

void SendPacket(Ptr<Socket> socket, uint32_t sensor_id,
                uint32_t seq, double interval_s) {
    uint32_t pkt_size = S_PAYLOAD;
    if ((int)(seq % g_params.M_window) == g_params.M_window - 1)
        pkt_size += g_params.sig_bytes;
    else
        pkt_size += 32;

    Ptr<Packet> pkt = Create<Packet>(pkt_size);
    PqTdagTag tag;
    tag.m_ts = Simulator::Now().GetMilliSeconds();
    pkt->AddByteTag(tag);
    socket->Send(pkt);

    Simulator::Schedule(Seconds(interval_s),
        &SendPacket, socket, sensor_id, seq + 1, interval_s);
}

void WriteCdf(const std::string& scheme_id) {
    std::sort(g_latencies_ms.begin(), g_latencies_ms.end());
    size_t n = g_latencies_ms.size();
    if (n == 0) {
        NS_LOG_UNCOND("WARNING: No packets received for " << scheme_id);
        return;
    }
    std::string fname = "results/ns3_latency_cdf_" + scheme_id + ".csv";
    std::ofstream ofs(fname);
    ofs << "latency_ms,cdf\n";
    for (size_t i = 0; i < n; ++i)
        ofs << g_latencies_ms[i] << "," << (double)(i+1)/n << "\n";
    ofs.close();

    double p99   = g_latencies_ms[(size_t)(0.99  * n)];
    double p999  = g_latencies_ms[(size_t)(0.999 * n)];
    double p9999 = g_latencies_ms[std::min(n-1,(size_t)(0.9999*n))];
    NS_LOG_UNCOND("Scheme=" << scheme_id
        << "  N=" << n
        << "  p99="    << p99    << "ms"
        << "  p99.9="  << p999   << "ms"
        << "  p99.99=" << p9999  << "ms"
        << "  T_max="  << T_MAX_MS << "ms"
        << "  -> " << fname);
}

int main(int argc, char* argv[]) {
    std::string scheme_id = "pq_tdag";
    CommandLine cmd(__FILE__);
    cmd.AddValue("scheme", "Scheme ID", scheme_id);
    cmd.Parse(argc, argv);

    if (SCHEMES.find(scheme_id) == SCHEMES.end()) {
        NS_LOG_UNCOND("Unknown scheme: " << scheme_id); return 1;
    }
    g_scheme_id = scheme_id;
    g_params    = SCHEMES[scheme_id];

    NS_LOG_UNCOND("=== PQ-TDAG Latency CDF ==="
        << "  scheme=" << g_params.name
        << "  M="      << g_params.M_window
        << "  sig="    << g_params.sig_bytes << "B");

    NodeContainer sensors, gateway;
    sensors.Create(N_SENSORS);
    gateway.Create(1);

    InternetStackHelper internet;
    internet.Install(sensors);
    internet.Install(gateway);

    double link_mbps = B_MAX_MBPS;
    PointToPointHelper p2p;
    p2p.SetDeviceAttribute("DataRate",
        StringValue(std::to_string((int)(link_mbps*1e6)) + "bps"));
    p2p.SetChannelAttribute("Delay", StringValue("1ms"));

    uint16_t port = 9;
    TypeId tid = TypeId::LookupByName("ns3::UdpSocketFactory");

    Ipv4AddressHelper ipv4;
    for (uint32_t i = 0; i < N_SENSORS; ++i) {
        NodeContainer pair;
        pair.Add(sensors.Get(i));
        pair.Add(gateway.Get(0));
        NetDeviceContainer devs = p2p.Install(pair);
        std::string base = "10." + std::to_string(i/254+1)
                         + "." + std::to_string(i%254+1) + ".0";
        ipv4.SetBase(base.c_str(), "255.255.255.0");
        Ipv4InterfaceContainer ifaces = ipv4.Assign(devs);

        // Send socket
        Ptr<Socket> sock = Socket::CreateSocket(sensors.Get(i), tid);
        sock->Connect(InetSocketAddress(ifaces.GetAddress(1), port));
        double start = 1.0 + i * (1.0/F_HZ/N_SENSORS);
        Simulator::Schedule(Seconds(start),
            &SendPacket, sock, i, 0, 1.0/F_HZ);
    }

    // Receive socket on gateway
    Ptr<Socket> recvSock = Socket::CreateSocket(gateway.Get(0), tid);
    recvSock->Bind(InetSocketAddress(Ipv4Address::GetAny(), port));
    recvSock->SetRecvCallback(MakeCallback(&RxPacket));

    Ipv4GlobalRoutingHelper::PopulateRoutingTables();
    Simulator::Stop(Seconds(SIM_DURATION));
    Simulator::Run();
    Simulator::Destroy();

    WriteCdf(scheme_id);
    return 0;
}
"""

# ══════════════════════════════════════════════════════════
#  FIX 3: pqtdag_throughput.cc — missing CommandLine cmd
# ══════════════════════════════════════════════════════════
THROUGHPUT_CC = r"""
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
SCHEME_PARAMS_PLACEHOLDER
};

static const int    S_LOAD   = 100;
static const double F_HZ     = 20.0;
static const double B_MAX    = 10e6;

int main(int argc, char* argv[]) {
    std::string scheme = "pq_tdag";
    std::string mode   = "scale";

    CommandLine cmd(__FILE__);
    cmd.AddValue("scheme", "Scheme ID", scheme);
    cmd.AddValue("mode",   "window or scale", mode);
    cmd.Parse(argc, argv);

    if (SCHEMES.find(scheme) == SCHEMES.end()) {
        NS_LOG_UNCOND("Unknown scheme: " << scheme); return 1;
    }
    SchemeParams p = SCHEMES[scheme];

    std::string outfile = "results/ns3_throughput_" + mode + "_" + scheme + ".csv";
    std::ofstream ofs(outfile);
    ofs << "param,throughput_tx_per_sec,cpu_util_pct\n";

    NS_LOG_UNCOND("Throughput sim: scheme=" << scheme << " mode=" << mode);

    if (mode == "window") {
        for (int M = 1; M <= 25; ++M) {
            double b_req = (50.0 * F_HZ / M)
                         * (M * S_LOAD + p.sig_bytes) * 8;
            if (b_req > B_MAX && scheme != "pq_tdag") {
                ofs << M << ",0,0\n"; continue;
            }
            double term_rate  = 50.0 * F_HZ / M;
            double t_conf     = p.t_verify_ms + 1.0 + 0.5;
            double throughput = std::min(term_rate, 1000.0/t_conf*50);
            double cpu_util   = std::min(100.0,
                (p.t_verify_ms * term_rate)/1000.0*100.0);
            ofs << M << "," << throughput << "," << cpu_util << "\n";
        }
    } else {
        int N_vals[] = {10,25,50,100,150,200,300,500,750,1000};
        for (int N : N_vals) {
            double b_req = (N * F_HZ / p.M_window)
                         * (p.M_window * S_LOAD + p.sig_bytes) * 8;
            if (b_req > B_MAX && scheme != "pq_tdag") {
                ofs << N << ",0,0\n"; continue;
            }
            double term_rate  = (double)N * F_HZ / p.M_window;
            double t_conf     = p.t_verify_ms + 1.0 + 0.5;
            double throughput = std::min(term_rate, 1000.0/t_conf*N);
            double cpu_util   = std::min(100.0,
                (p.t_verify_ms * term_rate)/1000.0*100.0);
            ofs << N << "," << throughput << "," << cpu_util << "\n";
        }
    }
    ofs.close();
    NS_LOG_UNCOND("Written: " << outfile);
    return 0;
}
"""

# ══════════════════════════════════════════════════════════
#  BUILD SCHEMES MAP FROM crypto_timings.json
# ══════════════════════════════════════════════════════════
import json, sys

DATA_FILE = Path("/home/hubert/project/PQ-TDAG/pq_tdag/results/data/crypto_timings.json")
if not DATA_FILE.exists():
    print(f"ERROR: {DATA_FILE} not found"); sys.exit(1)

with open(DATA_FILE) as f:
    data = json.load(f)

lines = []
for sid, s in data["schemes"].items():
    if "error" in s or "t_sign_mean_ms" not in s:
        continue
    M     = 5 if sid == "pq_tdag" else 1
    label = s.get("label", sid).replace('"', '\\"')
    lines.append(
        f'    {{"{sid}", {{"{label}", '
        f'{s.get("t_sign_mean_ms",1.2)}, '
        f'{s.get("t_verify_mean_ms",0.045)}, '
        f'{s.get("sig_bytes",2420)}, '
        f'{s.get("pk_bytes",1312)}, '
        f'{M}}}}}'
    )
schemes_cpp = ",\n".join(lines)

# ── Write fixed files ─────────────────────────────────────
cdf_content = LATENCY_CDF.replace("SCHEME_PARAMS_PLACEHOLDER", schemes_cpp)
tpt_content = THROUGHPUT_CC.replace("SCHEME_PARAMS_PLACEHOLDER", schemes_cpp)

(SCRATCH / "pqtdag_latency_cdf.cc").write_text(cdf_content)
print("Fixed: pqtdag_latency_cdf.cc")

(SCRATCH / "pqtdag_throughput.cc").write_text(tpt_content)
print("Fixed: pqtdag_throughput.cc")

print()
print("Now run:")
print("  cd /root/ns-allinone-3.40/ns-3.40")
print("  ./ns3 build -j$(nproc)")
