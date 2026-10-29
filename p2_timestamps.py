import numpy as np
import matplotlib.pyplot as plt

# --- Part 1: Spacing formula ---
def spacing(t):
    """Compute floating-point spacing at time t."""
    k = np.floor(np.log2(t))
    return 2.0**(k - 52)

# --- Part 2: Resolution in 2026 and 2150 ---
t_2026 = 1.782e9
t_2150 = 5.68e9

dt_2026 = spacing(t_2026)
dt_2150 = spacing(t_2150)

print("=== Timestamp Resolution ===")
print(f"2026: t = {t_2026:.3e} s, Δt = {dt_2026:.6e} s = {dt_2026*1e9:.2f} ns")
print(f"2150: t = {t_2150:.3e} s, Δt = {dt_2150:.6e} s = {dt_2150*1e9:.2f} ns")

# --- Part 3: Threshold for 1 microsecond ---
# Solve: 1e-6 < 0.5 * 2^(floor(log2(t)) - 52)
# floor(log2(t)) - 52 > log2(2e-6)
# floor(log2(t)) > 52 + log2(2e-6)
threshold_k = 52 + np.log2(2e-6)
t_thresh = 2.0**(np.ceil(threshold_k))
print(f"\n=== Threshold ===")
print(f"floor(log2(t)) > {threshold_k:.2f}")
print(f"t_thresh = 2^{np.ceil(threshold_k):.0f} = {t_thresh:.6e} s")
print(f"Approx year: {1970 + t_thresh/(365.25*24*3600):.0f}")

# --- Part 4: Accumulation simulation ---
def simulate_accumulation(step=1e-7, n_steps=10_000_000, use_kahan=False):
    """Accumulate step n_steps times, return final sum and error history."""
    t = 0.0
    errors = []
    exact = 0.0
    c = 0.0  # Kahan compensation
    
    for i in range(n_steps):
        exact += step
        if use_kahan:
            y = step - c
            t_new = t + y
            c = (t_new - t) - y
            t = t_new
        else:
            t += step
        
        if i % 1_000_000 == 0:
            errors.append(abs(t - exact))
    
    return t, exact, errors

# Naive accumulation
print("\n=== Naive Accumulation ===")
t_naive, exact, errors_naive = simulate_accumulation(step=1e-7, n_steps=10_000_000, use_kahan=False)
print(f"Final sum (naive): {t_naive:.10f}")
print(f"Exact sum:         {exact:.10f}")
print(f"Absolute error:    {abs(t_naive - exact):.6e}")
print(f"Relative error:    {abs(t_naive - exact)/exact:.6e}")

# Kahan accumulation
print("\n=== Kahan Accumulation ===")
t_kahan, exact, errors_kahan = simulate_accumulation(step=1e-7, n_steps=10_000_000, use_kahan=True)
print(f"Final sum (Kahan): {t_kahan:.10f}")
print(f"Exact sum:         {exact:.10f}")
print(f"Absolute error:    {abs(t_kahan - exact):.6e}")
print(f"Relative error:    {abs(t_kahan - exact)/exact:.6e}")

# Plot error growth
plt.figure(figsize=(10, 5))
steps = np.arange(len(errors_naive)) * 1_000_000
plt.plot(steps, errors_naive, 'r-o', label='Naive')
plt.plot(steps, errors_kahan, 'b-s', label='Kahan')
plt.xlabel('Steps')
plt.ylabel('Absolute Error')
plt.title('Accumulation Error: Naive vs Kahan')
plt.legend()
plt.grid(True)
plt.savefig('accumulation_error.png', dpi=150)
plt.show()