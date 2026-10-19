// Options market-making simulator.
//
// A market maker quotes a small book of index options around a fitted volatility surface,
// delta-hedges in futures, and faces two kinds of flow:
//   * uninformed flow, arriving at a rate that falls with quote distance from fair
//     (Avellaneda-Stoikov intensity  lambda = lambda0 * exp(-k (delta - h)) )
//   * informed flow, which sees fair value H steps ahead and trades only when that
//     future value is outside the quotes (adverse selection).
//
// Three quoting policies are compared on identical random numbers (common random numbers):
//   1  symmetric quotes at the market half-spread, no inventory control
//   2  quotes skewed by net vega inventory (reservation-price idea applied to vega),
//      with a hard vega limit
//   3  policy 2 plus adaptive width: widen when short-window realised vol runs above
//      implied, and when recent fills show negative markouts (toxic flow)
//
// Every step's P&L is attributed to edge, delta, gamma, theta, vega, vanna, volga,
// residual and hedging cost, and the attribution is reconciled against an independent
// cash-plus-positions wealth calculation.
//
// Usage:  mmsim <config file> <output per-path csv>
#include "bs.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <fstream>
#include <map>
#include <random>
#include <sstream>
#include <string>
#include <vector>

struct Instrument {
    bs::Type type;
    double K, T, r, q, vol, half_vol_pts, lambda0;  // lambda0 = fills per day per side at market width
};

struct Config {
    double S0 = 5500, mult = 10, atm_ref = 0.17;
    int steps_per_day = 78, days = 5, n_paths = 10000, policy = 1;
    unsigned long long seed = 20261012ULL;
    // underlying and vol-level dynamics
    double kappa = 4.0, eta = 0.20, rho_sv = -0.7, vrp = 0.025;
    // flow
    double k_mult = 1.0;            // decay k = k_mult * ln2 / h  (per vol point)
    double informed_frac = 0.10;    // share of baseline arrivals that are informed
    int horizon = 6;                // informed look-ahead, steps
    // policy parameters
    double vega_limit = 2000;       // EUR per vol point; skew reaches one half-spread here
    double hard_limit_mult = 2.5;   // policies 2-3 stop adding risk beyond this multiple
    double alpha = 1.0, beta = 0.5; // policy 3 widening weights
    int rv_window = 12, tox_window = 20;
    // hedging
    double fut_half_spread = 0.5;   // index points
    double hedge_bps = 0.0;         // extra cost in bp of notional
    double fut_mult = 10;
    int hedge_lag = 1;              // fills are hedged this many steps later (latency)
    // stress
    int stress_step = -1;
    double gap_move = 0.0, vol_shock = 0.0;
    std::vector<Instrument> inst;
};

static Config load(const char* path) {
    Config c;
    std::ifstream f(path);
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ss(line);
        std::string key;
        ss >> key;
        if (key == "inst") {
            std::string t;
            Instrument in{};
            ss >> t >> in.K >> in.T >> in.r >> in.q >> in.vol >> in.half_vol_pts >> in.lambda0;
            in.type = (t == "C") ? bs::Type::Call : bs::Type::Put;
            c.inst.push_back(in);
            continue;
        }
        double v;
        ss >> v;
        std::map<std::string, double*> d = {
            {"S0", &c.S0}, {"mult", &c.mult}, {"atm_ref", &c.atm_ref}, {"kappa", &c.kappa}, {"eta", &c.eta},
            {"rho_sv", &c.rho_sv}, {"vrp", &c.vrp}, {"k_mult", &c.k_mult}, {"informed_frac", &c.informed_frac},
            {"vega_limit", &c.vega_limit}, {"hard_limit_mult", &c.hard_limit_mult}, {"alpha", &c.alpha},
            {"beta", &c.beta}, {"fut_half_spread", &c.fut_half_spread}, {"hedge_bps", &c.hedge_bps},
            {"fut_mult", &c.fut_mult}, {"gap_move", &c.gap_move}, {"vol_shock", &c.vol_shock}};
        std::map<std::string, int*> di = {
            {"steps_per_day", &c.steps_per_day}, {"days", &c.days}, {"n_paths", &c.n_paths}, {"policy", &c.policy},
            {"horizon", &c.horizon}, {"rv_window", &c.rv_window}, {"tox_window", &c.tox_window},
            {"stress_step", &c.stress_step}, {"hedge_lag", &c.hedge_lag}};
        if (d.count(key)) *d[key] = v;
        else if (di.count(key)) *di[key] = (int)v;
        else if (key == "seed") c.seed = (unsigned long long)v;
        else { std::fprintf(stderr, "unknown key %s\n", key.c_str()); std::exit(1); }
    }
    return c;
}

struct PathResult {
    double total, edge, delta, gamma, theta, vega, vanna, volga, residual, cost, recon_err;
    double markout_inf, markout_unf, hmarkout_inf, hmarkout_unf, max_dd, max_abs_vega, mean_abs_vega;
    int fills, fills_inf;
};

static PathResult run_path(const Config& c, int p) {
    const int n_inst = (int)c.inst.size();
    const int N = c.steps_per_day * c.days;
    const double dt = 1.0 / (252.0 * c.steps_per_day);
    const double dt_days = 1.0 / c.steps_per_day;

    std::mt19937_64 rng(c.seed * 1000003ULL + (unsigned long long)p);
    std::normal_distribution<double> nrm(0.0, 1.0);
    std::uniform_real_distribution<double> uni(0.0, 1.0);

    // 1. pre-generate the market path (identical across policies for a given path id)
    std::vector<double> S(N + 1), x(N + 1);
    S[0] = c.S0; x[0] = 0.0;
    for (int j = 0; j < N; ++j) {
        const double z1 = nrm(rng), z2 = c.rho_sv * z1 + std::sqrt(1 - c.rho_sv * c.rho_sv) * nrm(rng);
        const double sig = std::max(c.atm_ref + x[j] - c.vrp, 0.03);
        S[j + 1] = S[j] * std::exp(-0.5 * sig * sig * dt + sig * std::sqrt(dt) * z1);
        x[j + 1] = x[j] - c.kappa * x[j] * dt + c.eta * std::sqrt(dt) * z2;
        if (j + 1 == c.stress_step) { S[j + 1] *= (1 + c.gap_move); x[j + 1] += c.vol_shock; }
    }
    // 2. pre-generate flow uniforms in a fixed order: [step][instrument][ask, bid, informed]
    std::vector<double> U((size_t)N * n_inst * 3);
    for (auto& u : U) u = uni(rng);

    // 3. fair values and Greeks on the path (sticky-strike surface plus parallel vol shift)
    std::vector<bs::Greeks> G((size_t)(N + 1) * n_inst);
    for (int j = 0; j <= N; ++j)
        for (int i = 0; i < n_inst; ++i) {
            const Instrument& in = c.inst[i];
            const double tau = in.T - j * dt;
            G[(size_t)j * n_inst + i] = bs::price_greeks(in.type, S[j], in.K, tau, in.r, in.q, std::max(in.vol + x[j], 0.02));
        }
    auto g = [&](int j, int i) -> const bs::Greeks& { return G[(size_t)j * n_inst + i]; };

    PathResult R{};
    std::vector<int> q(n_inst, 0);
    std::vector<std::vector<int>> qhist;   // positions at the end of each step (after fills)
    double h = 0.0, cash = 0.0, peak = 0.0, sum_abs_vega = 0.0;
    struct Fill { int step, inst, sgn; double half_px, delta; };
    std::deque<Fill> pending;        // fills waiting for their markout to mature
    std::deque<double> tox;          // matured markouts in half-spreads (policy 3)

    for (int j = 0; j < N; ++j) {
        // mature markouts (known to the desk only after H steps)
        while (!pending.empty() && pending.front().step + c.horizon <= j) {
            const Fill& f = pending.front();
            const int jl = std::min(f.step + c.hedge_lag, j);
            const double mo = f.sgn * ((g(j, f.inst).price - g(f.step, f.inst).price) - f.delta * (S[j] - S[jl]));
            tox.push_back(mo / f.half_px);
            if ((int)tox.size() > c.tox_window) tox.pop_front();
            pending.pop_front();
        }
        double vega_net = 0;   // EUR per 1 vol point
        for (int i = 0; i < n_inst; ++i) vega_net += q[i] * g(j, i).vega * 0.01 * c.mult;
        sum_abs_vega += std::fabs(vega_net);
        R.max_abs_vega = std::max(R.max_abs_vega, std::fabs(vega_net));

        double widen = 1.0;
        if (c.policy == 3) {
            const int w0 = std::max(0, j - c.rv_window);
            double ss = 0;
            for (int t = w0; t < j; ++t) { const double lr = std::log(S[t + 1] / S[t]); ss += lr * lr; }
            const double rv = (j > w0) ? std::sqrt(ss / ((j - w0) * dt)) : c.atm_ref;
            const double iv = c.atm_ref + x[j];
            double tx = 0;
            if (!tox.empty()) { for (double t : tox) tx += t; tx /= tox.size(); }
            widen = 1.0 + c.alpha * std::max(0.0, rv / iv - 1.0) + c.beta * std::max(0.0, -tx);
        }

        for (int i = 0; i < n_inst; ++i) {
            const Instrument& in = c.inst[i];
            const bs::Greeks& gi = g(j, i);
            const double vpp = gi.vega * 0.01;                       // price per vol point
            if (vpp < 1e-6) continue;
            const double h_vol = in.half_vol_pts;
            double skew_vol = 0;
            bool block_buy = false, block_sell = false;
            if (c.policy >= 2) {
                skew_vol = h_vol * std::clamp(vega_net / c.vega_limit, -2.0, 2.0);
                const double hard = c.hard_limit_mult * c.vega_limit;
                if (vega_net > hard) block_buy = true;                 // long vega: stop buying options
                if (vega_net < -hard) block_sell = true;
            }
            const double centre = gi.price - skew_vol * vpp;
            const double half = h_vol * widen * vpp;
            const double ask = centre + half, bid = std::max(centre - half, 0.0);
            const double d_ask = std::max((ask - gi.price) / vpp, 0.0);
            const double d_bid = std::max((gi.price - bid) / vpp, 0.0);
            const double k = c.k_mult * std::log(2.0) / h_vol;
            const double lam_ask = in.lambda0 * std::exp(-k * (d_ask - h_vol));
            const double lam_bid = in.lambda0 * std::exp(-k * (d_bid - h_vol));
            const double lam_inf = c.informed_frac / (1 - c.informed_frac) * 2 * in.lambda0;
            const double* u = &U[((size_t)j * n_inst + i) * 3];

            auto fill = [&](int sgn, double px, bool informed) {  // sgn +1: desk buys
                q[i] += sgn;
                cash -= sgn * px * c.mult;
                R.edge += sgn * (gi.price - px) * c.mult;
                R.fills++;
                const int jh = std::min(j + c.horizon, N);
                const double mo = sgn * (g(jh, i).price - gi.price) * c.mult;
                const int jlag = std::min(j + c.hedge_lag, jh);   // hedge only starts working after the lag
                const double hmo = sgn * ((g(jh, i).price - gi.price) - gi.delta * (S[jh] - S[jlag])) * c.mult;
                if (informed) { R.fills_inf++; R.markout_inf += mo; R.hmarkout_inf += hmo; }
                else { R.markout_unf += mo; R.hmarkout_unf += hmo; }
                pending.push_back({j, i, sgn, std::max(half, 1e-9), gi.delta});
            };
            if (!block_sell && u[0] < 1 - std::exp(-lam_ask * dt_days)) fill(-1, ask, false);
            if (!block_buy && u[1] < 1 - std::exp(-lam_bid * dt_days)) fill(+1, bid, false);
            if (u[2] < 1 - std::exp(-lam_inf * dt_days)) {
                const double fut = g(std::min(j + c.horizon, N), i).price;
                if (fut > ask && !block_sell) fill(-1, ask, true);
                else if (fut < bid && !block_buy) fill(+1, bid, true);
            }
        }

        // delta hedge in whole futures contracts
        qhist.push_back(q);
        const int jl = (int)qhist.size() - 1 - c.hedge_lag;
        const std::vector<int> zero(n_inst, 0);
        const std::vector<int>& qh = (jl >= 0) ? qhist[jl] : zero;
        double d_net = h;
        for (int i = 0; i < n_inst; ++i) d_net += qh[i] * g(j, i).delta * c.mult;
        const double trade = -std::round(d_net / c.fut_mult) * c.fut_mult;
        if (trade != 0) {
            const double cost = std::fabs(trade) * (c.fut_half_spread + S[j] * c.hedge_bps * 1e-4);
            h += trade;
            cash -= trade * S[j] + cost;
            R.cost += cost;
        }

        // step to j+1 and attribute
        const double dS = S[j + 1] - S[j], dx = x[j + 1] - x[j];
        double mtm = h * dS, de = h * dS, ga = 0, th = 0, ve = 0, va = 0, vo = 0;
        for (int i = 0; i < n_inst; ++i) {
            if (!q[i]) continue;
            const bs::Greeks& a = g(j, i);
            const double n = q[i] * c.mult;
            mtm += n * (g(j + 1, i).price - a.price);
            de += n * a.delta * dS;
            ga += 0.5 * n * a.gamma * dS * dS;
            th += n * a.theta * dt;
            ve += n * a.vega * dx;
            va += n * a.vanna * dS * dx;
            vo += 0.5 * n * a.volga * dx * dx;
        }
        R.delta += de; R.gamma += ga; R.theta += th; R.vega += ve; R.vanna += va; R.volga += vo;
        R.residual += mtm - (de + ga + th + ve + va + vo);

        double wealth = cash + h * S[j + 1];
        for (int i = 0; i < n_inst; ++i) wealth += q[i] * g(j + 1, i).price * c.mult;
        peak = std::max(peak, wealth);
        R.max_dd = std::max(R.max_dd, peak - wealth);
        R.total = wealth;
    }
    const double attributed = R.edge + R.delta + R.gamma + R.theta + R.vega + R.vanna + R.volga + R.residual - R.cost;
    R.recon_err = R.total - attributed;
    R.mean_abs_vega = sum_abs_vega / N;
    return R;
}

int main(int argc, char** argv) {
    if (argc < 3) { std::fprintf(stderr, "usage: mmsim config out.csv\n"); return 1; }
    const Config c = load(argv[1]);
    std::vector<PathResult> res(c.n_paths);
#pragma omp parallel for schedule(dynamic, 64)
    for (int p = 0; p < c.n_paths; ++p) res[p] = run_path(c, p);
    FILE* f = std::fopen(argv[2], "w");
    std::fprintf(f, "path,total,edge,delta,gamma,theta,vega,vanna,volga,residual,cost,recon_err,"
                    "markout_inf,markout_unf,hmarkout_inf,hmarkout_unf,max_dd,max_abs_vega,mean_abs_vega,fills,fills_inf\n");
    for (int p = 0; p < c.n_paths; ++p) {
        const PathResult& r = res[p];
        std::fprintf(f, "%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.3e,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%d,%d\n", p,
                     r.total, r.edge, r.delta, r.gamma, r.theta, r.vega, r.vanna, r.volga, r.residual, r.cost, r.recon_err,
                     r.markout_inf, r.markout_unf, r.hmarkout_inf, r.hmarkout_unf, r.max_dd, r.max_abs_vega, r.mean_abs_vega, r.fills, r.fills_inf);
    }
    std::fclose(f);
    return 0;
}
