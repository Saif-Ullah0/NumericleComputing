import numpy as np
import matplotlib.pyplot as plt

# --- Case Study 5: Trig Function Stability ---
print("=== CS5: Trig Function Stability ===")

x_test = 1e-7
direct = 1 - np.cos(x_test)
stable = 2 * np.sin(x_test/2)**2
true_val = 5e-15  # approximately

print(f"\nFor x = {x_test}:")
print(f"Direct:  1 - cos(x)     = {direct:.20e}")
print(f"Stable:  2*sin²(x/2)    = {stable:.20e}")
print(f"True:                    = {true_val:.20e}")
print(f"Relative error (direct): {abs(direct - true_val)/true_val:.6e}")
print(f"Relative error (stable): {abs(stable - true_val)/true_val:.6e}")

# --- Error vs x ---
x_vals = np.logspace(-12, -1, 500)

# True value using high precision
import mpmath as mp
mp.mp.dps = 50
true_vals = np.array([float(1 - mp.cos(mp.mpf(x))) for x in x_vals])

# Direct and stable
direct_vals = 1 - np.cos(x_vals)
stable_vals = 2 * np.sin(x_vals/2)**2

# Relative errors
rel_direct = np.abs(direct_vals - true_vals) / np.abs(true_vals)
rel_stable = np.abs(stable_vals - true_vals) / np.abs(true_vals)

# Plot
plt.figure(figsize=(10, 6))
plt.loglog(x_vals, rel_direct, 'r-', label='Direct: 1 - cos(x)')
plt.loglog(x_vals, rel_stable, 'b--', label='Stable: 2 sin²(x/2)')
plt.xlabel('x')
plt.ylabel('Relative Error')
plt.title('Relative Error: Direct vs Stable Evaluation of 1 - cos(x)')
plt.legend()
plt.grid(True, which='both', alpha=0.3)
plt.savefig('trig_stability.png', dpi=150)
plt.show()

print("\nPlot saved as trig_stability.png")