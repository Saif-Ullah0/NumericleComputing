import numpy as np
from function_hub import bisection, secant

# --- Case Study 1: Projectile Flight Time ---
f = lambda t: 3*t**2 - 18*t + 15

print("=== CS1: Projectile Flight Time ===")
print("f(t) = 3t^2 - 18t + 15 = 0\n")

# Root existence
print(f"f(0.5) = {f(0.5)}")
print(f"f(2.0) = {f(2.0)}")
print(f"f(0.5)*f(2.0) = {f(0.5)*f(2.0)} < 0 -> root in [0.5, 2.0]\n")

# --- Bisection with both stopping criteria ---
print("--- Bisection (criterion: |f(t)| < 1e-5) ---")
root_bi_f, iter_bi_f, hist_bi_f = bisection(f, 0.5, 2.0, tol=1e-5, criterion='f')
print(f"Root: {root_bi_f:.10f}, Iterations: {iter_bi_f}")

print("\n--- Bisection (criterion: |t_{k+1} - t_k| < 1e-5) ---")
root_bi_s, iter_bi_s, hist_bi_s = bisection(f, 0.5, 2.0, tol=1e-5, criterion='step')
print(f"Root: {root_bi_s:.10f}, Iterations: {iter_bi_s}")

# --- Secant with both stopping criteria ---
print("\n--- Secant (criterion: |f(t)| < 1e-5) ---")
root_se_f, iter_se_f, hist_se_f = secant(f, 0.5, 2.0, tol=1e-5, criterion='f')
print(f"Root: {root_se_f:.10f}, Iterations: {iter_se_f}")

print("\n--- Secant (criterion: |t_{k+1} - t_k| < 1e-5) ---")
root_se_s, iter_se_s, hist_se_s = secant(f, 0.5, 2.0, tol=1e-5, criterion='step')
print(f"Root: {root_se_s:.10f}, Iterations: {iter_se_s}")

# --- Comparison table ---
print("\n=== Comparison Table ===")
print(f"{'Method':12s} {'Criterion':10s} {'Root':>15s} {'Iterations':>10s}")
print(f"{'Bisection':12s} {'f':10s} {root_bi_f:15.10f} {iter_bi_f:10d}")
print(f"{'Bisection':12s} {'step':10s} {root_bi_s:15.10f} {iter_bi_s:10d}")
print(f"{'Secant':12s} {'f':10s} {root_se_f:15.10f} {iter_se_f:10d}")
print(f"{'Secant':12s} {'step':10s} {root_se_s:15.10f} {iter_se_s:10d}")

# Exact root
print(f"\nExact root: t = 1.0")