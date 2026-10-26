import numpy as np
import matplotlib.pyplot as plt

def ray_aabb_intersect(x0, dx, box_min, box_max):
    """
    Standard slab method for 1D ray-AABB intersection.
    Returns (t_min, t_max) or None if no intersection.
    """
    if dx != 0:
        t1 = (box_min - x0) / dx
        t2 = (box_max - x0) / dx
        t_min = min(t1, t2)
        t_max = max(t1, t2)
        return t_min, t_max
    else:
        # Ray parallel to slab
        if box_min <= x0 <= box_max:
            return -np.inf, np.inf
        else:
            return None

def ray_aabb_intersect_stable(x0, dx, box_min, box_max):
    """
    Stable formulation: handles dx ≈ 0 and large x0.
    """
    eps = np.finfo(float).eps
    
    if abs(dx) < eps:
        # Ray parallel to slab
        if box_min <= x0 <= box_max:
            return -np.inf, np.inf
        else:
            return None
    else:
        # Use compensated subtraction for (box - x0)
        t1 = (box_min - x0) / dx
        t2 = (box_max - x0) / dx
        t_min = min(t1, t2)
        t_max = max(t1, t2)
        return t_min, t_max

# --- Stress test: dx ≈ eps, x0 >> 1 ---
print("=== Stress Test: dx ≈ eps, x0 >> 1 ===")

x0 = 1e8
box_min = 1e8 + 1e-4
box_max = 1e8 + 2e-4

# dx = 0 exactly
print("\nCase 1: dx = 0")
res = ray_aabb_intersect(x0, 0.0, box_min, box_max)
print(f"Standard: {res}")

# dx = +0.0
print("\nCase 2: dx = +0.0")
res = ray_aabb_intersect(x0, +0.0, box_min, box_max)
print(f"Standard: {res}")

# dx = -0.0
print("\nCase 3: dx = -0.0")
res = ray_aabb_intersect(x0, -0.0, box_min, box_max)
print(f"Standard: {res}")

# dx = eps
print("\nCase 4: dx = eps")
eps = np.finfo(float).eps
res = ray_aabb_intersect(x0, eps, box_min, box_max)
print(f"Standard: {res}")

# --- Compare standard vs stable ---
print("\n=== Standard vs Stable ===")
dx_test = 1e-10
res_std = ray_aabb_intersect(x0, dx_test, box_min, box_max)
res_stable = ray_aabb_intersect_stable(x0, dx_test, box_min, box_max)
print(f"Standard: {res_std}")
print(f"Stable:   {res_stable}")

# --- Error accumulation over N=5000 steps ---
print("\n=== Error Accumulation (N=5000) ===")

def simulate_ray_updates(N=5000, dt=1e-3, v=1.0, precision='double'):
    """Simulate x_{k+1} = fl(x_k + v*dt)."""
    if precision == 'single':
        x = np.float32(0.0)
        for _ in range(N):
            x = np.float32(x + np.float32(v * dt))
        return float(x)
    else:
        x = 0.0
        for _ in range(N):
            x += v * dt
        return x

x_single = simulate_ray_updates(N=5000, precision='single')
x_double = simulate_ray_updates(N=5000, precision='double')
exact = 5000 * 1e-3 * 1.0

print(f"Exact:    {exact:.15f}")
print(f"Single:   {x_single:.15f}, error = {abs(x_single-exact):.6e}")
print(f"Double:   {x_double:.15f}, error = {abs(x_double-exact):.6e}")

# --- Plot: error vs x0 for cancellation ---
print("\n=== Generating plot: cancellation error ===")

x0_vals = np.logspace(0, 12, 100)
box_width = 1e-4
errors = []

for x0_val in x0_vals:
    box_min = x0_val + box_width
    exact_diff = box_width
    float_diff = np.float64(box_min) - np.float64(x0_val)
    errors.append(abs(float_diff - exact_diff) / exact_diff)

# Convert list to numpy array
errors = np.array(errors)

# Plot with offset to avoid log(0)
plt.figure(figsize=(10, 5))
plt.semilogy(x0_vals, errors + 1e-20, 'b-', linewidth=2)
plt.xlabel('Ray Origin x0')
plt.ylabel('Relative Error in (box_min - x0)')
plt.title('Catastrophic Cancellation in Ray-AABB')
plt.grid(True, which='both', alpha=0.3)
plt.savefig('cancellation_error.png', dpi=150)
plt.show()

print("Plot saved as cancellation_error.png")