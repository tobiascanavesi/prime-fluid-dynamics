#!/usr/bin/env python3
"""
Reproduce all results from:
"A Burgers-Fisher Equation for Prime Gaps: Spectral Proof of Convergence to Poisson"

Tobias Canavesi, April 2026

Generates:
  - Symbolic verification of Theorems 2.1, 3.1
  - MI decay rate (Figure 1a)
  - Information cascade (Figure 1b)
  - Sieve diffusion model (Figure 1c)
  - Velocity field visualization (Figure 1d)
"""

import numpy as np
import time
from pathlib import Path
from collections import Counter
from math import log, log2, exp
from sympy import nextprime, factorint, Symbol, diff, exp as sp_exp, simplify, Function
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import linregress
from scipy.special import eval_laguerre, eval_genlaguerre

output_dir = Path(__file__).parent
N_PRIMES = 200_000


def generate_primes_near(target, count):
    primes = []
    p = nextprime(target - 1)
    for _ in range(count):
        primes.append(p)
        p = nextprime(p)
    return np.array(primes)


def shannon_entropy(counts_dict, total):
    return -sum((c / total) * log2(c / total) for c in counts_dict.values() if c > 0)


def mi_modp(gaps_list, p):
    n = len(gaps_list) - 1
    joint = Counter()
    mx, my = Counter(), Counter()
    for i in range(n):
        rx, ry = gaps_list[i] % p, gaps_list[i + 1] % p
        joint[(rx, ry)] += 1
        mx[rx] += 1
        my[ry] += 1
    Hx = shannon_entropy(mx, n)
    Hy = shannon_entropy(my, n)
    Hj = shannon_entropy(joint, n)
    bc = (len(joint) - len(mx) - len(my) + 1) / (2 * n * log(2))
    return max(0.0, Hx + Hy - Hj - bc)


def mi_full(gaps_list):
    n = len(gaps_list) - 1
    joint = Counter()
    mx, my = Counter(), Counter()
    for i in range(n):
        joint[(gaps_list[i], gaps_list[i + 1])] += 1
        mx[gaps_list[i]] += 1
        my[gaps_list[i + 1]] += 1
    Hx = shannon_entropy(mx, n)
    Hy = shannon_entropy(my, n)
    Hj = shannon_entropy(joint, n)
    bc = (len(joint) - len(mx) - len(my) + 1) / (2 * n * log(2))
    return max(0.0, Hx + Hy - Hj - bc)


# ===========================================================================
print("=" * 70)
print("  SYMBOLIC VERIFICATION OF THEOREMS")
print("=" * 70)

v = Symbol('v', positive=True)


def apply_L(f_expr):
    """Fokker-Planck operator: L f = v*f'' + (v+1)*f' + f"""
    return v * diff(f_expr, v, 2) + (v + 1) * diff(f_expr, v) + f_expr


def apply_T(h_expr):
    """Linearized velocity operator: T h = v*h'' + (2-v)*h' - h"""
    return v * diff(h_expr, v, 2) + (2 - v) * diff(h_expr, v) - h_expr


# Theorem 2.1: Lf = lambda*f for f = e^{-v}*L_n(v)
print("\nTheorem 2.1 (Fokker-Planck eigenfunctions):")
laguerre = [1, 1 - v, 1 - 2*v + v**2/2, 1 - 3*v + 3*v**2/2 - v**3/6]
for n, Ln in enumerate(laguerre):
    f = sp_exp(-v) * Ln
    result = simplify(apply_L(f) - (-n) * f)
    print(f"  n={n}: L(e^{{-v}}*L_{n}) + {n}*e^{{-v}}*L_{n} = {result}")

# Theorem 3.1: Velocity eigenfunctions (associated Laguerre)
print("\nTheorem 3.1 (Velocity perturbation eigenfunctions):")
assoc_laguerre = [1, 2 - v, (v**2 - 6*v + 6) / 2]
for n, Ln1 in enumerate(assoc_laguerre):
    result = simplify(apply_T(Ln1) - (-(n + 1)) * Ln1)
    print(f"  n={n}: T(L_{n}^{{(1)}}) + {n+1}*L_{n}^{{(1)}} = {result}")

# Proposition 5.3: Mod-3 MI
print("\nProposition 5.3 (Mod-3 MI):")
P = {}
for a in [1, 2]:
    for b in [1, 2]:
        for c in [1, 2]:
            g1 = (b - a) % 3
            g2 = (c - b) % 3
            P[(g1, g2)] = P.get((g1, g2), 0) + 1/8

marg_r = {r: sum(P.get((r, s), 0) for s in range(3)) for r in range(3)}
marg_c = {s: sum(P.get((r, s), 0) for r in range(3)) for s in range(3)}
mi3 = sum(P[(r, s)] * log2(P[(r, s)] / (marg_r[r] * marg_c[s]))
          for (r, s) in P if P[(r, s)] > 0 and marg_r[r] * marg_c[s] > 0)
print(f"  MI_3 = {mi3:.6f} bits (expected: 0.250000)")
print(f"  P(1,1) = {P.get((1,1), 0)}, P(2,2) = {P.get((2,2), 0)} (expected: 0)")


# ===========================================================================
print("\n" + "=" * 70)
print("  NUMERICAL EXPERIMENTS")
print("=" * 70)

SCALES = [10**3, 3*10**3, 10**4, 3*10**4, 10**5, 3*10**5,
          10**6, 3*10**6, 10**7, 3*10**7, 10**8]

prime_data = {}
print("\nGenerating primes...")
for scale in SCALES:
    t0 = time.time()
    primes = generate_primes_near(scale, N_PRIMES)
    gaps = np.diff(primes)
    log_x = log(float(np.median(primes)))
    prime_data[scale] = (primes, gaps, log_x)
    print(f"  {scale:.0e}: log(x)={log_x:.2f} ({time.time()-t0:.1f}s)")

# MI decay
print("\nMI decay rate:")
mi_data = []
for scale in SCALES:
    _, gaps, log_x = prime_data[scale]
    gl = gaps.tolist()
    mt = mi_full(gl)
    m3 = mi_modp(gl, 3)
    mi_data.append((log_x, mt, m3))
    print(f"  {scale:.0e}: MI_total={mt:.4f}, MI_3={m3:.4f}, MI_3/total={m3/mt*100:.1f}%")

logx = np.array([d[0] for d in mi_data])
mi_t = np.array([d[1] for d in mi_data])
mi_3 = np.array([d[2] for d in mi_data])
sl, it, r, _, se = linregress(np.log(logx), np.log(mi_t))
print(f"\n  FIT: MI_total ~ {exp(it):.3f} / (log x)^{{{-sl:.3f}}}, R^2={r**2:.4f}")


# ===========================================================================
print("\n" + "=" * 70)
print("  GENERATING FIGURES")
print("=" * 70)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.patch.set_facecolor('#0f172a')
for ax in axes.flat:
    ax.set_facecolor('#1e293b')
    ax.tick_params(colors='#94a3b8')
    ax.xaxis.label.set_color('#94a3b8')
    ax.yaxis.label.set_color('#94a3b8')
    ax.title.set_color('#e2e8f0')
    for spine in ax.spines.values():
        spine.set_color('#334155')
    ax.grid(True, alpha=0.2, color='#475569')

# (a) MI decay
ax = axes[0, 0]
ax.plot(logx, mi_t, 'o-', color='#38bdf8', markersize=5, label='MI total')
ax.plot(logx, mi_3, 's-', color='#f472b6', markersize=4, label='MI at p=3')
x_fit = np.linspace(logx.min(), logx.max(), 100)
ax.plot(x_fit, exp(it) * x_fit**sl, '--', color='#38bdf8', alpha=0.4,
        label=f'fit: $(\log x)^{{{sl:.2f}}}$')
ax.set_xlabel('log(x)')
ax.set_ylabel('MI (bits)')
ax.set_title('(a) Mutual Information Decay')
ax.legend(fontsize=8, facecolor='#1e293b', edgecolor='#334155', labelcolor='#e2e8f0')

# (b) Information cascade
ax = axes[0, 1]
scale_108 = SCALES[-1]
_, gaps_108, _ = prime_data[scale_108]
gl_108 = gaps_108.tolist()
sieve_primes = [3, 5, 7, 11, 13, 17, 19, 23]
mi_by_p = [mi_modp(gl_108, p) for p in sieve_primes]
ax.bar(range(len(sieve_primes)), mi_by_p, color='#a78bfa', alpha=0.8)
ax.set_xticks(range(len(sieve_primes)))
ax.set_xticklabels([str(p) for p in sieve_primes])
ax.set_xlabel('Sieve prime p')
ax.set_ylabel('MI(g mod p, g\' mod p) bits')
ax.set_title('(b) Information Cascade at 10^8')

# (c) Velocity field at different scales
ax = axes[1, 0]
v_grid = np.linspace(0.05, 5, 200)
for scale in [SCALES[0], SCALES[4], SCALES[-1]]:
    _, gaps, log_x = prime_data[scale]
    bins = np.linspace(0, 6, 80)
    hist, _ = np.histogram(gaps / log_x, bins=bins, density=True)
    v_c = 0.5 * (bins[:-1] + bins[1:])
    # u = -f'/f approximated by -d(log f)/dv
    log_f = np.log(np.maximum(hist, 1e-10))
    u = -np.gradient(log_f, v_c[1] - v_c[0])
    label = f'$10^{{{int(np.log10(scale))}}}$'
    ax.plot(v_c[5:-5], u[5:-5], '-', alpha=0.7, linewidth=1.2, label=label)
ax.axhline(y=1, color='#fbbf24', linestyle='--', linewidth=2, alpha=0.6, label='u=1 (Poisson)')
ax.set_xlabel('v = g / log(x)')
ax.set_ylabel('u(v) = -d log f / dv')
ax.set_ylim(-1, 4)
ax.set_title('(c) Velocity Field: Converging to u=1')
ax.legend(fontsize=8, facecolor='#1e293b', edgecolor='#334155', labelcolor='#e2e8f0')

# (d) Sieve diffusion: adding primes
ax = axes[1, 1]
scale_fit = 10**7
_, gaps_fit, log_x_fit = prime_data[scale_fit]
L = log_x_fit
max_g = int(10 * L)
gv = np.arange(2, max_g + 1, 2)

P_cramer = np.exp(-gv / L)
P_cramer /= P_cramer.sum()

def ss_factor(g, p):
    return (p - 1) / (p - 2) if g % p == 0 else 1.0

emp_counts = Counter(gaps_fit.tolist())
P_emp = np.array([emp_counts.get(g, 0) for g in gv], dtype=float)
P_emp /= P_emp.sum()

sieve_seq = [3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
P = P_cramer.copy()
kl_vals = []
kl_0 = sum(p * log(p / q) for p, q in zip(P_emp, P_cramer) if p > 0 and q > 0)
kl_vals.append(kl_0)

for sp in sieve_seq:
    for i, g in enumerate(gv):
        P[i] *= ss_factor(int(g), sp)
    P /= P.sum()
    kl = sum(p * log(p / q) for p, q in zip(P_emp, P) if p > 0 and q > 0)
    kl_vals.append(kl)

labels = ['None'] + [str(p) for p in sieve_seq]
colors_bar = ['#475569'] + ['#f472b6' if i == 0 else '#a78bfa' if i < 3 else '#334155'
              for i in range(len(sieve_seq))]
ax.bar(range(len(kl_vals)), kl_vals, color=colors_bar, alpha=0.8)
ax.set_xticks(range(0, len(labels), 2))
ax.set_xticklabels([labels[i] for i in range(0, len(labels), 2)], fontsize=7)
ax.set_xlabel('Largest sieve prime')
ax.set_ylabel('KL(empirical || model) nats')
ax.set_title('(d) Sieve Diffusion at 10^7')

fig.suptitle('A Burgers-Fisher Equation for Prime Gaps', fontsize=16,
             fontweight='bold', color='#e2e8f0', y=1.01)
fig.tight_layout()
fig.savefig(output_dir / 'figure1.pdf', dpi=200, bbox_inches='tight',
            facecolor='#0f172a')
fig.savefig(output_dir / 'figure1.png', dpi=200, bbox_inches='tight',
            facecolor='#0f172a')
plt.close(fig)

print(f"\nSaved: figure1.pdf, figure1.png")
print(f"Total runtime: {time.time() - t0:.0f}s")
print("\nDone.")
