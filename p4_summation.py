import numpy as np
import matplotlib.pyplot as plt
import time

# --- Generate synthetic array ---
np.random.seed(42)
N = 10_000_000

# Create array with disparate scales
arr = np.zeros(N)
arr[0] = 1e9
arr[1] = -1e9
arr[2:] = np.random.uniform(-1e-7, 1e-7, N-2)

# --- Summation methods ---
def naive_sum(arr):
    """Left-to-right sequential sum."""
    s = 0.0
    for x in arr:
        s += x
    return s

def sorted_sum(arr):
    """Sum smallest to largest magnitude."""
    return sum(sorted(arr, key=abs))

def pairwise_sum(arr):
    """Pairwise (tree) summation."""
    if len(arr) <= 1:
        return arr[0] if len(arr) == 1 else 0.0
    mid = len(arr) // 2
    return pairwise_sum(arr[:mid]) + pairwise_sum(arr[mid:])

def kahan_sum(arr):
    """Kahan compensated summation."""
    s = 0.0
    c = 0.0
    for x in arr:
        y = x - c
        t = s + y
        c = (t - s) - y
        s = t
    return s

# --- Exact baseline using mpmath ---
try:
    import mpmath as mp
    mp.mp.dps = 50
    exact = mp.mpf(0)
    for x in arr:
        exact += mp.mpf(x)
    exact = float(exact)
except ImportError:
    # Fallback: use math.fsum (exact)
    import math
    exact = math.fsum(arr)

print(f"Exact sum: {exact:.15e}")

# --- Benchmark ---
methods = {
    'Naive': naive_sum,
    'Sorted': sorted_sum,
    'Pairwise': pairwise_sum,
    'Kahan': kahan_sum,
}

results = {}
for name, method in methods.items():
    start = time.time()
    result = method(arr)
    elapsed = time.time() - start
    error = abs(result - exact)
    results[name] = {
        'sum': result,
        'error': error,
        'time': elapsed,
    }
    print(f"{name:10s}: sum = {result:.15e}, error = {error:.6e}, time = {elapsed:.4f}s")

# --- Comparison table ---
print("\n=== Comparison Table ===")
print(f"{'Method':12s} {'Sum':>20s} {'Abs Error':>15s} {'Time (s)':>10s}")
for name, data in results.items():
    print(f"{name:12s} {data['sum']:20.15e} {data['error']:15.6e} {data['time']:10.4f}")

# --- Plot error comparison ---
names = list(results.keys())
errors = [results[n]['error'] for n in names]

plt.figure(figsize=(8, 5))
plt.bar(names, errors, color=['red', 'orange', 'blue', 'green'])
plt.yscale('log')
plt.ylabel('Absolute Error')
plt.title('Summation Error Comparison (N = 10^7)')
plt.grid(True, axis='y')
plt.savefig('summation_error.png', dpi=150)
plt.show()