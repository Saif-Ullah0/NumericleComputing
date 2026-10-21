import numpy as np
from function_hub import regula_falsi, bisection

# --- Case Study 3: Queue Delay Model ---
f = lambda D: D - 50*(1 - np.exp(-0.03*D)) - 10

print("=== CS3: Queue Delay Model ===")
print("f(D) = D - 50(1 - e^(-0.03D)) - 10 = 0\n")

# Root existence
print(f"f(10) = {f(10):.5f}")
print(f"f(60) = {f(60):.5f}")
print(f"f(10)*f(60) = {f(10)*f(60):.5f} < 0 -> root in [10, 60]\n")

# --- Regula Falsi ---
print("--- Regula Falsi ---")
root_rf, iter_rf, hist_rf = regula_falsi(f, 10, 60, tol=1e-5, criterion='f')
print(f"Root: {root_rf:.10f}, Iterations: {iter_rf}")

# --- Bisection ---
print("\n--- Bisection ---")
root_bi, iter_bi, hist_bi = bisection(f, 10, 60, tol=1e-5, criterion='f')
print(f"Root: {root_bi:.10f}, Iterations: {iter_bi}")

# --- Comparison table ---
print("\n=== Comparison Table ===")
print(f"{'Method':15s} {'Root':>15s} {'Iterations':>10s}")
print(f"{'Regula Falsi':15s} {root_rf:15.10f} {iter_rf:10d}")
print(f"{'Bisection':15s} {root_bi:15.10f} {iter_bi:10d}")

# Comment
print("\nComment: Regula Falsi converges faster but Bracketing is guaranteed in both.")
print("For real-time network simulators, Bisection is more predictable (known iteration count).")