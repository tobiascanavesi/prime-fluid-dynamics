# Paste this into https://sagecell.sagemath.org/ for independent verification
# Paper: "A Burgers-Fisher Equation for Prime Gaps"

var('v')

# FP operator: L f = v*f'' + (v+1)*f' + f
def L(f):
    return v*diff(f,v,2) + (v+1)*diff(f,v) + f

# Velocity perturbation operator: T h = v*h'' + (2-v)*h' - h
def T(h):
    return v*diff(h,v,2) + (2-v)*diff(h,v) - h

all_pass = True

print("=== Theorem 2.1: FP eigenfunctions ===")
print("Claim: L(e^{-v}*L_n(v)) = -n * e^{-v}*L_n(v)")
for n in range(6):
    Ln = laguerre(n, v)
    f = exp(-v)*Ln
    residual = (L(f) + n*f).expand().simplify()
    status = "PASS" if residual == 0 else "FAIL"
    if residual != 0: all_pass = False
    print(f"  n={n}: [{status}] residual = {residual}")

print()
print("=== Theorem 3.1: Velocity eigenfunctions ===")
print("Claim: T(L_n^{(1)}(v)) = -(n+1) * L_n^{(1)}(v)")
for n in range(5):
    Ln1 = gen_laguerre(n, 1, v)
    residual = (T(Ln1) + (n+1)*Ln1).expand().simplify()
    status = "PASS" if residual == 0 else "FAIL"
    if residual != 0: all_pass = False
    print(f"  n={n}: [{status}] residual = {residual}")

print()
print("=== Lemma 2.1: Stationary solution ===")
residual = L(exp(-v)).expand().simplify()
status = "PASS" if residual == 0 else "FAIL"
if residual != 0: all_pass = False
print(f"  [{status}] L(e^(-v)) = {residual}")

print()
print("=== Theorem 2.3: Equilibrium check ===")
print(f"  [PASS] u=1: u*(1-u) = {1*(1-1)}")

print()
print("=== Theorem 2.1(ii): Substitution reduces to Laguerre equation ===")
h = function('h')(v)
f = exp(-v)*h
Lf_over_emv = (L(f) / exp(-v)).expand().simplify()
expected = v*diff(h,v,2) + (1-v)*diff(h,v)
residual = (Lf_over_emv - expected).expand().simplify()
status = "PASS" if residual == 0 else "FAIL"
if residual != 0: all_pass = False
print(f"  [{status}] L(e^(-v)*h)/e^(-v) - [v*h''+(1-v)*h'] = {residual}")

print()
print("=== Proposition 5.3: Mod-3 MI ===")
from itertools import product as cartprod
P = {}
for a,b,c in cartprod([1,2], repeat=3):
    g1 = (b-a)%3
    g2 = (c-b)%3
    P[(g1,g2)] = P.get((g1,g2), 0) + QQ(1)/8
mr = {r: sum(P.get((r,s),0) for s in range(3)) for r in range(3)}
mc = {s: sum(P.get((r,s),0) for r in range(3)) for s in range(3)}
MI = sum(p*log(p/(mr[r]*mc[s]),2) for (r,s),p in P.items()
         if p > 0 and mr[r]*mc[s] > 0)

status_mi = "PASS" if MI == QQ(1)/4 else "FAIL"
status_z1 = "PASS" if P.get((1,1),0) == 0 else "FAIL"
status_z2 = "PASS" if P.get((2,2),0) == 0 else "FAIL"
if MI != QQ(1)/4 or P.get((1,1),0) != 0 or P.get((2,2),0) != 0:
    all_pass = False
print(f"  [{status_mi}] MI = {MI} bits (expected 1/4)")
print(f"  [{status_z1}] P(1,1) = {P.get((1,1),0)} (expected 0)")
print(f"  [{status_z2}] P(2,2) = {P.get((2,2),0)} (expected 0)")

print()
print("=" * 50)
if all_pass:
    print("ALL VERIFICATIONS PASSED.")
else:
    print("SOME VERIFICATIONS FAILED.")
print("=" * 50)
