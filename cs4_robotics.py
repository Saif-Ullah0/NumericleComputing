import numpy as np
from function_hub import newton, secant

# --- Case Study 4: Robotics Inverse Kinematics ---
f = lambda th: np.cos(th) - 0.4*th
df = lambda th: -np.sin(th) - 0.4

print("=== CS4: Robotics Inverse Kinematics ===")
print("f(θ) = cos(θ) - 0.4θ = 0\n")

# Root existence and uniqueness
print(f"f(0.5) = {f(0.5):.6f}")
print(f"f(1.5) = {f(1.5):.6f}")
print(f"f(0.5)*f(1.5) = {f(0.5)*f(1.5):.6f} < 0 -> root exists\n")

# Show derivative is always negative
print(f"f'(θ) = -sin(θ) - 0.4")
print(f"f'(0.5) = {df(0.5):.6f}")
print(f"f'(1.5) = {df(1.5):.6f}")
print("→ f'(θ) < 0 for all θ in [0.5, 1.5] → unique root\n")

# --- Newton's Method ---
print("--- Newton's Method ---")
root_nw, iter_nw, hist_nw = newton(f, df, 1.0, tol=1e-6, criterion='f')
print(f"Root: {root_nw:.10f}, Iterations: {iter_nw}")

# --- Secant Method ---
print("\n--- Secant Method ---")
root_se, iter_se, hist_se = secant(f, 0.5, 1.5, tol=1e-6, criterion='f')
print(f"Root: {root_se:.10f}, Iterations: {iter_se}")

# --- Comparison table ---
print("\n=== Comparison Table ===")
print(f"{'Method':15s} {'Root':>15s} {'Iterations':>10s}")
print(f"{'Newton':15s} {root_nw:15.10f} {iter_nw:10d}")
print(f"{'Secant':15s} {root_se:15.10f} {iter_se:10d}")

# Comment
print("\nComment: Newton's method is preferred in robotic control loops")
print("because it converges quadratically and f'(θ) is easy to compute analytically.")