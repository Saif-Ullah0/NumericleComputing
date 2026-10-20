import numpy as np
from function_hub import fixed_point, newton

# --- Case Study 2: Softmax Threshold ---
f = lambda x: 1/(1 + np.exp(-2*x)) - 0.9
g = lambda x: x - 2*(1/(1 + np.exp(-2*x)) - 0.9)
df = lambda x: 2*np.exp(-2*x) / (1 + np.exp(-2*x))**2

x_star = 0.5 * np.log(9)
print("=== CS2: Softmax Threshold ===")
print(f"True root: x* = {x_star:.10f}\n")

# --- Convergence check ---
g_prime = lambda x: 1 - 4*np.exp(-2*x)/(1 + np.exp(-2*x))**2
print(f"|g'(x*)| = {abs(g_prime(x_star)):.6f} < 1 -> converges\n")

# --- Fixed-Point ---
print("--- Fixed-Point Iteration ---")
root_fp, iter_fp, hist_fp = fixed_point(g, 1.0, tol=1e-5, criterion='step')
print(f"Root: {root_fp:.10f}, Iterations: {iter_fp}")
print(f"Error: {abs(root_fp - x_star):.6e}")

# --- Newton ---
print("\n--- Newton's Method ---")
root_nw, iter_nw, hist_nw = newton(f, df, 1.0, tol=1e-5, criterion='f')
print(f"Root: {root_nw:.10f}, Iterations: {iter_nw}")
print(f"Error: {abs(root_nw - x_star):.6e}")

# --- Comparison table ---
print("\n=== Comparison Table ===")
print(f"{'Method':15s} {'Root':>15s} {'Iterations':>10s} {'Error':>12s}")
print(f"{'Fixed-Point':15s} {root_fp:15.10f} {iter_fp:10d} {abs(root_fp-x_star):12.6e}")
print(f"{'Newton':15s} {root_nw:15.10f} {iter_nw:10d} {abs(root_nw-x_star):12.6e}")

print(f"\nTrue root: x* = {x_star:.10f}")