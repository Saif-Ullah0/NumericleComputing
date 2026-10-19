from math import isfinite
from numbers import Integral, Real


def _validate_options(tol, max_iter, criterion):
    if isinstance(tol, bool) or not isinstance(tol, Real) or not isfinite(tol) or tol <= 0:
        raise ValueError("tol must be a finite positive number")
    if isinstance(max_iter, bool) or not isinstance(max_iter, Integral) or max_iter <= 0:
        raise ValueError("max_iter must be a positive integer")
    if not isinstance(criterion, str) or criterion not in ("f", "step", "rel"):
        raise ValueError("criterion must be 'f', 'step', or 'rel'")

def bisection(f, a, b, tol=1e-6, max_iter=100, criterion='f'):
    _validate_options(tol, max_iter, criterion)
    if f(a) * f(b) > 0:
        raise ValueError("f(a) and f(b) must have opposite signs")
    history = []
    x_prev = None
    for i in range(1, max_iter + 1):
        p = (a + b) / 2.0
        fp = f(p)
        history.append((i, p, fp))
        if criterion == 'f':
            if abs(fp) < tol:
                return p, i, history
        elif criterion == 'step':
            if x_prev is not None and abs(p - x_prev) < tol:
                return p, i, history
        elif criterion == 'rel':
            if x_prev is not None and abs(p - x_prev) / abs(p) < tol:
                return p, i, history
        if f(a) * fp < 0:
            b = p
        else:
            a = p
        x_prev = p
    return p, max_iter, history

def fixed_point(g, x0, tol=1e-6, max_iter=100, criterion='f'):
    _validate_options(tol, max_iter, criterion)
    history = []
    x = x0
    for i in range(1, max_iter + 1):
        x_new = g(x)
        history.append((i, x_new, x_new - x))
        if criterion == 'f':
            if abs(x_new - x) < tol:
                return x_new, i, history
        elif criterion == 'step':
            if abs(x_new - x) < tol:
                return x_new, i, history
        elif criterion == 'rel':
            if abs(x_new - x) / abs(x_new) < tol:
                return x_new, i, history
        x = x_new
    return x, max_iter, history

def newton(f, df, x0, tol=1e-6, max_iter=100, criterion='f'):
    _validate_options(tol, max_iter, criterion)
    history = []
    x = x0
    for i in range(1, max_iter + 1):
        fx = f(x)
        dfx = df(x)
        if dfx == 0:
            raise ValueError("Derivative is zero")
        x_new = x - fx / dfx
        history.append((i, x_new, f(x_new)))
        if criterion == 'f':
            if abs(f(x_new)) < tol:
                return x_new, i, history
        elif criterion == 'step':
            if abs(x_new - x) < tol:
                return x_new, i, history
        elif criterion == 'rel':
            if abs(x_new - x) / abs(x_new) < tol:
                return x_new, i, history
        x = x_new
    return x, max_iter, history

def regula_falsi(f, x0, x1, tol=1e-6, max_iter=100, criterion='f'):
    _validate_options(tol, max_iter, criterion)
    if f(x0) * f(x1) > 0:
        raise ValueError("f(x0) and f(x1) must have opposite signs")
    history = []
    q0 = f(x0)
    q1 = f(x1)
    for i in range(1, max_iter + 1):
        p = x1 - q1 * (x1 - x0) / (q1 - q0)
        q = f(p)
        history.append((i, p, q))
        if criterion == 'f':
            if abs(q) < tol:
                return p, i, history
        elif criterion == 'step':
            if abs(p - x1) < tol:
                return p, i, history
        elif criterion == 'rel':
            if abs(p - x1) / abs(p) < tol:
                return p, i, history
        if q * q1 < 0:
            x0 = x1
            q0 = q1
        x1 = p
        q1 = q
    return p, max_iter, history

def secant(f, x0, x1, tol=1e-6, max_iter=100, criterion='f'):
    _validate_options(tol, max_iter, criterion)
    history = []
    q0 = f(x0)
    q1 = f(x1)
    for i in range(1, max_iter + 1):
        if q1 - q0 == 0:
            raise ValueError("Division by zero in secant")
        p = x1 - q1 * (x1 - x0) / (q1 - q0)
        q = f(p)
        history.append((i, p, q))
        if criterion == 'f':
            if abs(q) < tol:
                return p, i, history
        elif criterion == 'step':
            if abs(p - x1) < tol:
                return p, i, history
        elif criterion == 'rel':
            if abs(p - x1) / abs(p) < tol:
                return p, i, history
        x0, q0 = x1, q1
        x1, q1 = p, q
    return p, max_iter, history

# --- Quick test ---
if __name__ == "__main__":
    f = lambda x: x**2 - 2
    df = lambda x: 2*x
    g = lambda x: (x + 2/x) / 2

    print("=== Function Hub Test: f(x) = x^2 - 2 ===\n")
    root, iters, _ = bisection(f, 1, 2, tol=1e-6, criterion='f')
    print(f"Bisection:     root = {root:.10f}, iters = {iters}")
    root, iters, _ = fixed_point(g, 1.5, tol=1e-6, criterion='step')
    print(f"Fixed-Point:   root = {root:.10f}, iters = {iters}")
    root, iters, _ = newton(f, df, 1.5, tol=1e-6, criterion='f')
    print(f"Newton:        root = {root:.10f}, iters = {iters}")
    root, iters, _ = regula_falsi(f, 1, 2, tol=1e-6, criterion='f')
    print(f"Regula Falsi:  root = {root:.10f}, iters = {iters}")
    root, iters, _ = secant(f, 1, 2, tol=1e-6, criterion='f')
    print(f"Secant:        root = {root:.10f}, iters = {iters}")