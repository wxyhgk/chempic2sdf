# 2026-10-02 chemical-head trust-region trial

The bounded trial was executed after a fresh one-shot authorization. It used
the fixed 1,200 released training samples, trained only
`pair.chemical.{weight,bias}`, and added initial-head probability-space KL
distillation with weight `0.5`.

Run: `outputs/experiments/chemvector-chemical-head-probe-20261002-trust05/`.

Runtime was healthy: RTX 5090/CUDA, 1,200 attempts, 1,198 optimizer updates,
2 empty-supervision skips, about 463 seconds. No sealed test was read and no
default/deployment artifact was changed.

| checkpoint | full exact | curriculum exact | full fallback | new fallback |
|---:|---:|---:|---:|---:|
| initial | 170 | 63 | 51 | 0 |
| 400 | 170 | 63 | 50 | 1 |
| 800 | 171 | 64 | 49 | 1 |
| 1200 | 171 | 64 | 51 | 3 |

There were no broken exact cases. The final rescued cases were the same two
seen in r3, but the final arm still introduced
`001401_coordgen`, `001833_original`, and `001866_stress` as fallbacks.

Decision: **NO-GO; keep initial as production baseline.** The trust-region
reduced some intermediate drift but did not satisfy the zero-new-fallback
gate. Do not spend another full 1,200-step run on a scalar trust weight. The
next change must target per-edge supervision/decoder boundary handling and be
validated first on the three fixed fallback witnesses.
