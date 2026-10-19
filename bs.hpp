// Black-Scholes-Merton pricer for European options: prices, analytic Greeks,
// implied volatility (Newton-Raphson with bisection fallback) and a Monte Carlo
// cross-check. Continuous dividend yield q, risk-free rate r, year fractions.
#pragma once
#include <cmath>
#include <cstdint>

namespace bs {

enum class Type { Call, Put };

struct Greeks {
    double price, delta, gamma, vega, theta, rho;  // vega per 1.00 vol, theta per year
    double vanna, volga;                           // d(delta)/d(vol), d(vega)/d(vol)
};

inline double norm_pdf(double x) { return 0.3989422804014327 * std::exp(-0.5 * x * x); }
inline double norm_cdf(double x) { return 0.5 * std::erfc(-x * 0.7071067811865476); }

inline Greeks price_greeks(Type t, double S, double K, double T, double r, double q, double vol) {
    const double sqT = std::sqrt(T);
    const double d1 = (std::log(S / K) + (r - q + 0.5 * vol * vol) * T) / (vol * sqT);
    const double d2 = d1 - vol * sqT;
    const double dfq = std::exp(-q * T), dfr = std::exp(-r * T);
    const double pdf1 = norm_pdf(d1);
    Greeks g{};
    g.gamma = dfq * pdf1 / (S * vol * sqT);
    g.vega = S * dfq * pdf1 * sqT;
    g.vanna = -dfq * pdf1 * d2 / vol;
    g.volga = g.vega * d1 * d2 / vol;
    if (t == Type::Call) {
        const double Nd1 = norm_cdf(d1), Nd2 = norm_cdf(d2);
        g.price = S * dfq * Nd1 - K * dfr * Nd2;
        g.delta = dfq * Nd1;
        g.theta = -S * dfq * pdf1 * vol / (2 * sqT) - r * K * dfr * Nd2 + q * S * dfq * Nd1;
        g.rho = K * T * dfr * Nd2;
    } else {
        const double Nmd1 = norm_cdf(-d1), Nmd2 = norm_cdf(-d2);
        g.price = K * dfr * Nmd2 - S * dfq * Nmd1;
        g.delta = -dfq * Nmd1;
        g.theta = -S * dfq * pdf1 * vol / (2 * sqT) + r * K * dfr * Nmd2 - q * S * dfq * Nmd1;
        g.rho = -K * T * dfr * Nmd2;
    }
    return g;
}

inline double price(Type t, double S, double K, double T, double r, double q, double vol) {
    return price_greeks(t, S, K, T, r, q, vol).price;
}

struct IVResult { double vol; int iters; bool ok; };

// Newton-Raphson on vega, falling back to bisection when a step leaves the bracket
// or vega is too small (deep in/out of the money).
inline IVResult implied_vol(Type t, double target, double S, double K, double T, double r, double q,
                            double tol = 1e-12, int max_iter = 100) {
    double lo = 1e-6, hi = 5.0, v = 0.3;
    for (int i = 1; i <= max_iter; ++i) {
        const Greeks g = price_greeks(t, S, K, T, r, q, v);
        const double diff = g.price - target;
        if (std::fabs(diff) < tol) return {v, i, true};
        if (diff > 0) hi = v; else lo = v;
        double next = (g.vega > 1e-10) ? v - diff / g.vega : 0.5 * (lo + hi);
        if (!(next > lo && next < hi)) next = 0.5 * (lo + hi);
        v = next;
    }
    return {v, max_iter, false};
}

// Simple xorshift RNG + Box-Muller, so the Monte Carlo check needs no dependencies.
struct Rng {
    uint64_t s;
    explicit Rng(uint64_t seed) : s(seed ? seed : 88172645463325252ULL) {}
    double uniform() {
        s ^= s << 13; s ^= s >> 7; s ^= s << 17;
        return ((s >> 11) + 0.5) * (1.0 / 9007199254740992.0);
    }
    double normal() {
        const double u1 = uniform(), u2 = uniform();
        return std::sqrt(-2.0 * std::log(u1)) * std::cos(6.283185307179586 * u2);
    }
};

struct MCResult { double price, std_err; };

// Antithetic Monte Carlo under the risk-neutral measure.
inline MCResult mc_price(Type t, double S, double K, double T, double r, double q, double vol,
                         int n_pairs, uint64_t seed) {
    Rng rng(seed);
    const double drift = (r - q - 0.5 * vol * vol) * T, diff = vol * std::sqrt(T), df = std::exp(-r * T);
    double sum = 0, sum2 = 0;
    for (int i = 0; i < n_pairs; ++i) {
        const double z = rng.normal();
        const double s1 = S * std::exp(drift + diff * z), s2 = S * std::exp(drift - diff * z);
        double p1, p2;
        if (t == Type::Call) { p1 = std::fmax(s1 - K, 0.0); p2 = std::fmax(s2 - K, 0.0); }
        else { p1 = std::fmax(K - s1, 0.0); p2 = std::fmax(K - s2, 0.0); }
        const double x = 0.5 * (p1 + p2) * df;
        sum += x; sum2 += x * x;
    }
    const double mean = sum / n_pairs;
    const double var = sum2 / n_pairs - mean * mean;
    return {mean, std::sqrt(var / n_pairs)};
}

}  // namespace bs
