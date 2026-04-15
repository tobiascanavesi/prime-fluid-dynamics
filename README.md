# A Burgers-Fisher Equation for Prime Gaps

We prove that the velocity field $u = -\partial_v \log f$, derived from the prime gap density, satisfies a Burgers-Fisher equation in a porous medium whose unique stable equilibrium is the Poisson distribution.

> **A Burgers-Fisher Equation for Prime Gaps: Spectral Analysis of Convergence to Poisson**
> Tobias Canavesi, April 2026

## The Equation

$$\partial_t u + 2v\,u\,\partial_v u = v\,\partial_v^2 u + (v+2)\,\partial_v u + u(1-u)$$

- **Burgers nonlinearity** $2vu\,u_v$: the Navier-Stokes quadratic term
- **Porous medium diffusion** $v\,u_{vv}$: permeability = gap size
- **Fisher-KPP reaction** $u(1-u)$: self-corrects toward $u = 1$ (Poisson)

The Fisher-KPP term emerges naturally from the Fokker-Planck structure.

## Key Results

1. **Spectral gap = 1**: linearized perturbations have associated Laguerre polynomial eigenfunctions with eigenvalues $-(n+1)$. Every mode decays exponentially.

2. **MI decay rate**: $\text{MI}(x) \sim 0.555 / (\log x)^{0.186}$ ($R^2 = 0.986$). First quantitative convergence rate for prime gaps to Poisson.

3. **Prime 3 carries 94%** of the sieve signal. Primes 7+ decay as $(\log x)^{-3.5}$ or faster.

4. **Mod-3 MI = 1/4 bit exactly**, with structural zeros P(1,1) = P(2,2) = 0.

## Reproduce

```bash
pip install numpy sympy scipy matplotlib
python reproduce.py
```

Outputs: symbolic verification of all theorems, numerical experiments at 11 scales, and Figure 1 (4-panel).

## Files

| File | Description |
|------|-------------|
| `paper.tex` / `paper.pdf` | The paper (6 pages) |
| `reproduce.py` | Single script: all proofs + all figures |
| `interactive.html` | Visual explanation (open in browser) |
| `figure1.pdf` | Generated figure |

## Citation
Waiting to be in arxiv so the correct cite will be updated 
in the following days.
```bibtex
@article{canavesi2026burgersfisher,
  title={A Burgers-Fisher Equation for Prime Gaps: Spectral Analysis of Convergence to Poisson},
  author={Canavesi, Tobias},
  year={2026}
}
```

## License

MIT
