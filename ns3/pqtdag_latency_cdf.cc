/*
 * pqtdag_latency_cdf.cc
 * ─────────────────────────────────────────────────────────────────
 * NS-3.40 — PQ-TDAG Latency CDF Simulation
 * Measures end-to-end confirmation latency for all 8 schemes.
 *
 * Scheme parameters are injected from liboqs measurements
 * (i9-14900KF, liboqs 0.15.0, 500 iterations each).
 *
 * Build:
 *   cd ~/ns-allinone-3.40/ns-3.40
 *   ./ns3 run "pqtdag_pqtdag_latency_cdf --scheme=pq_tdag"
 *
 * Output: results/ns3_latency_cdf_{scheme}.csv
 *
 * Fixes applied vs generated version:
 *   [1] PqTdagTag replaces TimestampTag (conflict with ns3::TimestampTag)
 *   [2] SetRecvCallback uses Callback<void,Ptr<Socket>> (NS-3.40 API)
 *   [3] Packet received via socket->RecvFrom() inside callback
 * ─────────────────────────────────────────────────────────────────
 */

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

// ── Scheme parameters (measured on i9-14900KF, liboqs 0.15.0) ──
struct SchemeParams {
    std::string name;
    double      t_sign_ms;
    double      t_verify_ms;
    uint32_t    sig_bytes;
    uint32_t    pk_bytes;
    int         M_window;
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

static const uint32_t N_SENSORS    = 50;
static const double   F_HZ         = 20.0;
static const uint32_t S_PAYLOAD    = 100;
static const double   T_MAX_MS     = 50.0;
static const double   SIM_DURATION = 30.0;
static const double   B_MAX_MBPS   = 10.0;

std::vector<double> g_latencies_ms;
std::string         g_scheme_id;
SchemeParams        g_params;

// ── Fix [1]: renamed to PqTdagTag to avoid ns3::TimestampTag conflict ──
class PqTdagTag : public Tag {
public:
    static TypeId GetTypeId() {
        static TypeId tid = TypeId("PqTdagTag")
            .SetParent<Tag>()
            .AddConstructor<PqTdagTag>();
        return tid;
    }
    TypeId GetInstanceTypeId() const override { return GetTypeId(); }
    uint32_t GetSerializedSize()  const override { return 8; }
    void Serialize(TagBuffer buf) const override { buf.WriteDouble(m_ts); }
    void Deserialize(TagBuffer buf)    override  { m_ts = buf.ReadDouble(); }
    void Print(std::ostream& os) const override  { os << "ts=" << m_ts; }
    double m_ts{0.0};
};

// ── Fix [2]: correct NS-3.40 callback signature ──────────────────
void RxPacket(Ptr<Socket> socket) {
    Ptr<Packet> pkt;
    Address from;
    // Fix [3]: retrieve packets via RecvFrom loop
    while ((pkt = socket->RecvFrom(from))) {
        PqTdagTag tag;
        if (pkt->FindFirstMatchingByteTag(tag)) {
            double now_ms  = Simulator::Now().GetMilliSeconds();
            double lat_ms  = now_ms - tag.m_ts;
            double t_crypto = g_params.t_verify_ms
                            + g_params.t_sign_ms / g_params.M_window;
            double jitter   = ((double)rand() / RAND_MAX - 0.5) * 0.2;
            g_latencies_ms.push_back(lat_ms + t_crypto + std::abs(jitter));
        }
    }
}

void SendPacket(Ptr<Socket> socket, uint32_t sensor_id,
                uint32_t seq, double interval_s) {
    uint32_t pkt_size = S_PAYLOAD;
    if ((int)(seq % g_params.M_window) == g_params.M_window - 1)
        pkt_size += g_params.sig_bytes;
    else
        pkt_size += 32;  // hash link only

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
        NS_LOG_UNCOND("WARNING: no packets received for " << scheme_id);
        return;
    }

    std::string fname = "results/ns3_latency_cdf_" + scheme_id + ".csv";
    std::ofstream ofs(fname);
    ofs << "latency_ms,cdf\n";
    for (size_t i = 0; i < n; ++i)
        ofs << g_latencies_ms[i] << "," << (double)(i + 1) / n << "\n";
    ofs.close();

    double p99   = g_latencies_ms[(size_t)(0.99   * n)];
    double p999  = g_latencies_ms[(size_t)(0.999  * n)];
    double p9999 = g_latencies_ms[std::min(n-1,(size_t)(0.9999 * n))];
    NS_LOG_UNCOND("scheme=" << scheme_id
        << "  N=" << n
        << "  p99="    << p99    << "ms"
        << "  p99.9="  << p999   << "ms"
        << "  p99.99=" << p9999  << "ms"
        << "  file="   << fname);
}

int main(int argc, char* argv[]) {
    std::string scheme_id = "pq_tdag";
    CommandLine cmd(__FILE__);
    cmd.AddValue("scheme", "Scheme ID (pq_tdag|naive_mldsa44|...)", scheme_id);
    cmd.Parse(argc, argv);

    if (SCHEMES.find(scheme_id) == SCHEMES.end()) {
        NS_LOG_UNCOND("Unknown scheme: " << scheme_id
            << ". Available: pq_tdag naive_mldsa44 mldsa65 falcon512 "
               "slhdsa128s slhdsa128f xmssmt ecdsa");
        return 1;
    }
    g_scheme_id = scheme_id;
    g_params    = SCHEMES[scheme_id];

    NS_LOG_UNCOND("=== PQ-TDAG Latency CDF ==="
        << "  scheme=" << g_params.name
        << "  M=" << g_params.M_window
        << "  sig=" << g_params.sig_bytes << "B");

    NodeContainer sensors, gateway;
    sensors.Create(N_SENSORS);
    gateway.Create(1);

    InternetStackHelper internet;
    internet.Install(sensors);
    internet.Install(gateway);

    PointToPointHelper p2p;
    p2p.SetDeviceAttribute("DataRate",
        StringValue(std::to_string((int)(B_MAX_MBPS * 1e6)) + "bps"));
    p2p.SetChannelAttribute("Delay", StringValue("1ms"));

    uint16_t port = 9;
    TypeId   tid  = TypeId::LookupByName("ns3::UdpSocketFactory");

    Ipv4AddressHelper ipv4;
    for (uint32_t i = 0; i < N_SENSORS; ++i) {
        NodeContainer pair;
        pair.Add(sensors.Get(i));
        pair.Add(gateway.Get(0));
        NetDeviceContainer devs = p2p.Install(pair);

        std::string base = "10." + std::to_string(i / 254 + 1)
                         + "." + std::to_string(i % 254 + 1) + ".0";
        ipv4.SetBase(base.c_str(), "255.255.255.0");
        Ipv4InterfaceContainer ifaces = ipv4.Assign(devs);

        Ptr<Socket> sock = Socket::CreateSocket(sensors.Get(i), tid);
        sock->Connect(InetSocketAddress(ifaces.GetAddress(1), port));
        double start = 1.0 + i * (1.0 / F_HZ / N_SENSORS);
        Simulator::Schedule(Seconds(start),
            &SendPacket, sock, i, 0, 1.0 / F_HZ);
    }

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
