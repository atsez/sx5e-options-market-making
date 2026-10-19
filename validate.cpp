// Validation and benchmark harness for bs.hpp.
//  1. Put-call parity across a strike x expiry x vol grid
//  2. Analytic Greeks against central finite differences
//  3. Implied-vol round trip (price -> vol -> price)
//  4. Monte Carlo price against the closed form
//  5. Throughput: prices with full Greeks per second
// Also writes grid.csv so the Python simulator can be cross-checked against C++.
#include "bs.hpp"
#include <chrono>
#include <cstdio>
#include <vector>
#include <algorithm>

using namespace bs;

int main() {
    const double S = 100.0, r = 0.04, q = 0.01;
    std::vector<double> strikes, expiries, vols;
    for (double k = 60; k <= 140.0001; k += 5) strikes.push_back(k);            // 17 strikes
    for (double d : {7, 14, 30, 60, 91, 182, 273, 365, 547, 730}) expiries.push_back(d / 365.0);  // 10 expiries
    for (double v = 0.10; v <= 0.800001; v += 0.10) vols.push_back(v);          // 8 vols

    // 1-3: grid checks
    double max_parity = 0, max_greek_rel = 0, max_iv_err = 0;
    long n = 0, iv_n = 0, iv_fail = 0, iv_iters = 0;
    FILE* csv = std::fopen("grid.csv", "w");
    std::fprintf(csv, "type,K,T,vol,price,delta,gamma,vega,theta,rho\n");
    for (double K : strikes) for (double T : expiries) for (double v : vols) {
        const Greeks c = price_greeks(Type::Call, S, K, T, r, q, v);
        const Greeks p = price_greeks(Type::Put, S, K, T, r, q, v);
        std::fprintf(csv, "C,%.4f,%.17g,%.4f,%.15e,%.15e,%.15e,%.15e,%.15e,%.15e\n", K, T, v, c.price, c.delta, c.gamma, c.vega, c.theta, c.rho);
        std::fprintf(csv, "P,%.4f,%.17g,%.4f,%.15e,%.15e,%.15e,%.15e,%.15e,%.15e\n", K, T, v, p.price, p.delta, p.gamma, p.vega, p.theta, p.rho);
        n += 2;
        // parity: C - P = S e^{-qT} - K e^{-rT}
        max_parity = std::max(max_parity, std::fabs((c.price - p.price) - (S * std::exp(-q * T) - K * std::exp(-r * T))));

        // finite-difference Greeks on the call, Richardson-extrapolated central differences
        // with relative steps; skip contracts whose value is numerically negligible
        if (c.price > 1e-4) {
            auto P = [&](double s, double t, double rr, double vv) { return price(Type::Call, s, K, t, rr, q, vv); };
            auto D = [&](double s, double t, double rr, double vv) { return price_greeks(Type::Call, s, K, t, rr, q, vv).delta; };
            auto V = [&](double s, double t, double rr, double vv) { return price_greeks(Type::Call, s, K, t, rr, q, vv).vega; };
            // first derivative of f at x with step h, Richardson: (4 D(h/2) - D(h)) / 3
            auto d1r = [](auto f, double x, double h) {
                const double a = (f(x + h) - f(x - h)) / (2 * h);
                const double b = (f(x + h / 2) - f(x - h / 2)) / h;
                return (4 * b - a) / 3;
            };
            const double hS = 1e-3 * S, hv = 1e-3 * v, hT = 1e-3 * T, hr = 1e-4;
            const double fd_delta = d1r([&](double x) { return P(x, T, r, v); }, S, hS);
            const double fd_gamma = d1r([&](double x) { return D(x, T, r, v); }, S, hS);
            const double fd_vega  = d1r([&](double x) { return P(S, T, r, x); }, v, hv);
            const double fd_theta = -d1r([&](double x) { return P(S, x, r, v); }, T, hT);
            const double fd_rho   = d1r([&](double x) { return P(S, T, x, v); }, r, hr);
            const double fd_vanna = d1r([&](double x) { return D(S, T, r, x); }, v, hv);
            const double fd_volga = d1r([&](double x) { return V(S, T, r, x); }, v, hv);
            auto rel = [](double a, double b) { return std::fabs(a - b) / std::max(std::fabs(b), 1e-2); };
            max_greek_rel = std::max({max_greek_rel, rel(c.delta, fd_delta), rel(c.gamma, fd_gamma),
                                      rel(c.vega, fd_vega), rel(c.theta, fd_theta), rel(c.rho, fd_rho),
                                      rel(c.vanna, fd_vanna), rel(c.volga, fd_volga)});
        }
        // implied vol round trip on out-of-the-money side (how quotes are actually inverted)
        const Type t = (K >= S) ? Type::Call : Type::Put;
        const double target = (t == Type::Call) ? c.price : p.price;
        const double intrinsic_ish = 1e-8;
        if (target > intrinsic_ish) {
            const IVResult iv = implied_vol(t, target, S, K, T, r, q);
            ++iv_n; iv_iters += iv.iters;
            if (!iv.ok) ++iv_fail; else max_iv_err = std::max(max_iv_err, std::fabs(iv.vol - v));
        }
    }
    std::fclose(csv);

    // 4: Monte Carlo cross-check, 1y ATM call at 20% vol
    const Greeks atm = price_greeks(Type::Call, S, 100, 1.0, r, q, 0.20);
    const MCResult mc = mc_price(Type::Call, S, 100, 1.0, r, q, 0.20, 1000000, 42);

    // 5: throughput
    const int reps = 5'000'000;
    volatile double sink = 0;
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < reps; ++i) {
        const double K = 60 + (i % 81);
        sink += price_greeks(Type::Call, S, K, 0.25 + (i % 7) * 0.1, r, q, 0.15 + (i % 11) * 0.03).delta;
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    const double secs = std::chrono::duration<double>(t1 - t0).count();

    std::printf("contracts_priced=%ld\n", n);
    std::printf("max_put_call_parity_error=%.3e\n", max_parity);
    std::printf("max_greek_rel_error_vs_finite_diff=%.3e\n", max_greek_rel);
    std::printf("iv_cases=%ld iv_failures=%ld avg_iters=%.2f max_iv_error=%.3e\n", iv_n, iv_fail, (double)iv_iters / iv_n, max_iv_err);
    std::printf("atm_1y_call closed_form=%.6f mc=%.6f mc_std_err=%.6f z=%.2f\n", atm.price, mc.price, mc.std_err, (mc.price - atm.price) / mc.std_err);
    std::printf("throughput_price_plus_greeks_per_sec=%.0f\n", reps / secs);
    return 0;
}
