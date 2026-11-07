"""Parte IV: optimización con restricciones y programación lineal."""

from __future__ import annotations

from itertools import combinations
from pathlib import Path
import csv
import math

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs" / "parte_iv"


def write_csv(path: Path, header: list[str], rows: list[list[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def solve_lagrange() -> dict[str, float]:
    """IV.A.1 y IV.A.2: solución simbólica y sensibilidad."""
    x, y, lam = sp.symbols("x y lambda", real=True)
    C = x**2 + 2*y**2
    lagrangiana = C - lam*(x + y - 6)

    equations = [
        sp.diff(lagrangiana, x),
        sp.diff(lagrangiana, y),
        sp.diff(lagrangiana, lam),
    ]
    sol = sp.solve(equations, (x, y, lam), dict=True)[0]

    x_star = sp.simplify(sol[x])
    y_star = sp.simplify(sol[y])
    lam_star = sp.simplify(sol[lam])
    c_star = sp.simplify(C.subs(sol))

    # Verificación por sustitución x = 6 - y.
    C_sub = sp.expand(C.subs(x, 6-y))
    y_sub = sp.solve(sp.diff(C_sub, y), y)[0]
    x_sub = sp.simplify(6-y_sub)
    assert sp.simplify(x_sub-x_star) == 0
    assert sp.simplify(y_sub-y_star) == 0

    # Restricción general x + y = b.
    b = sp.symbols("b", real=True)
    y_b = b/3
    x_b = 2*b/3
    V_b = sp.simplify(x_b**2 + 2*y_b**2)
    dV_b = sp.diff(V_b, b)

    estimate_61 = sp.N(c_star + lam_star*sp.Rational(1, 10), 15)
    exact_61 = sp.N(V_b.subs(b, sp.Rational(61, 10)), 15)

    return {
        "x_star": float(x_star),
        "y_star": float(y_star),
        "lambda_star": float(lam_star),
        "C_star": float(c_star),
        "V_6_1_est": float(estimate_61),
        "V_6_1_exact": float(exact_61),
        "V_formula_coeff": float(sp.Rational(2, 3)),
        "dV_at_6": float(dV_b.subs(b, 6)),
    }


def plot_lagrange(result: dict[str, float]) -> Path:
    """IV.A.3: curvas de nivel, restricción y óptimo."""
    x_vals = np.linspace(-1, 7, 400)
    y_vals = np.linspace(-1, 7, 400)
    X, Y = np.meshgrid(x_vals, y_vals)
    C = X**2 + 2*Y**2

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.contour(X, Y, C, levels=[8, 16, 24, 32, 48, 72])
    ax.plot(x_vals, 6-x_vals, label="Restricción: x + y = 6")
    ax.scatter([result["x_star"]], [result["y_star"]], label="Óptimo (4, 2)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("IV.A — Multiplicadores de Lagrange")
    ax.set_xlim(-1, 7)
    ax.set_ylim(-1, 7)
    ax.legend()
    ax.grid(True)

    out = OUTPUT / "IV_A_lagrange.png"
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def feasible(point: np.ndarray, tol: float = 1e-9) -> bool:
    x, y = point
    return (
        x + y <= 4 + tol
        and x <= 2 + tol
        and y <= 3 + tol
        and x >= -tol
        and y >= -tol
    )


def enumerate_lp_vertices() -> list[tuple[float, float]]:
    """IV.B.1: enumera intersecciones de fronteras y conserva las factibles."""
    # Cada frontera se representa como a*x + b*y = c.
    boundaries = [
        np.array([1.0, 1.0, 4.0]),  # x + y = 4
        np.array([1.0, 0.0, 2.0]),  # x = 2
        np.array([0.0, 1.0, 3.0]),  # y = 3
        np.array([1.0, 0.0, 0.0]),  # x = 0
        np.array([0.0, 1.0, 0.0]),  # y = 0
    ]

    points: list[np.ndarray] = []
    for b1, b2 in combinations(boundaries, 2):
        A = np.array([[b1[0], b1[1]], [b2[0], b2[1]]], dtype=float)
        rhs = np.array([b1[2], b2[2]], dtype=float)
        if abs(np.linalg.det(A)) < 1e-12:
            continue
        p = np.linalg.solve(A, rhs)
        if feasible(p):
            if not any(np.allclose(p, q, atol=1e-9) for q in points):
                points.append(p)

    centroid = np.mean(points, axis=0)
    points.sort(key=lambda p: math.atan2(p[1]-centroid[1], p[0]-centroid[0]))
    return [(float(p[0]), float(p[1])) for p in points]


def solve_lp() -> dict[str, object]:
    """IV.B.2 y IV.B.3: evalúa objetivo y determina el máximo."""
    vertices = enumerate_lp_vertices()
    values = [(x, y, 3*x + 2*y) for x, y in vertices]
    optimum = max(values, key=lambda row: row[2])

    # Unicidad: un solo vértice alcanza el valor máximo.
    best = [row for row in values if np.isclose(row[2], optimum[2])]
    unique = len(best) == 1

    return {"vertices": vertices, "values": values, "optimum": optimum, "unique": unique}


def plot_lp(result: dict[str, object]) -> Path:
    """IV.B.1: región factible, vértices y recta de nivel óptima."""
    vertices = np.array(result["vertices"], dtype=float)
    optimum = result["optimum"]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.fill(vertices[:, 0], vertices[:, 1], alpha=0.25, label="Región factible")
    ax.plot(
        np.r_[vertices[:, 0], vertices[0, 0]],
        np.r_[vertices[:, 1], vertices[0, 1]],
        label="Frontera factible",
    )
    ax.scatter(vertices[:, 0], vertices[:, 1], label="Vértices")

    for x, y in vertices:
        ax.annotate(f"({x:g}, {y:g})", (x, y), xytext=(5, 5), textcoords="offset points")

    xs = np.linspace(0, 4, 200)
    ys = (10 - 3*xs)/2
    ax.plot(xs, ys, label="Recta óptima: 3x + 2y = 10")

    ax.scatter([optimum[0]], [optimum[1]], label="Óptimo")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("IV.B — Programación lineal")
    ax.set_xlim(-0.25, 4.25)
    ax.set_ylim(-0.25, 4.25)
    ax.legend()
    ax.grid(True)

    out = OUTPUT / "IV_B_region_factible.png"
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)

    print("\n=== PARTE IV ===")

    lagrange = solve_lagrange()
    assert np.isclose(lagrange["x_star"], 4)
    assert np.isclose(lagrange["y_star"], 2)
    assert np.isclose(lagrange["lambda_star"], 8)
    assert np.isclose(lagrange["C_star"], 24)

    print("\nIV.A — Multiplicadores de Lagrange")
    print(f"x* = {lagrange['x_star']:.6f}")
    print(f"y* = {lagrange['y_star']:.6f}")
    print(f"lambda* = {lagrange['lambda_star']:.6f}")
    print(f"C* = {lagrange['C_star']:.6f}")
    print(f"Estimación V(6.1) = {lagrange['V_6_1_est']:.6f}")
    print(f"Valor exacto V(6.1) = {lagrange['V_6_1_exact']:.6f}")
    plot_lagrange(lagrange)

    lp = solve_lp()
    print("\nIV.B — Programación lineal")
    for x, y, z in lp["values"]:
        print(f"({x:.6f}, {y:.6f}) -> z = {z:.6f}")
    ox, oy, oz = lp["optimum"]
    print(f"Óptimo: ({ox:.6f}, {oy:.6f}), z* = {oz:.6f}")
    print(f"Óptimo único: {lp['unique']}")

    write_csv(
        OUTPUT / "IV_B_vertices_objetivo.csv",
        ["x", "y", "z=3x+2y"],
        [[x, y, z] for x, y, z in lp["values"]],
    )
    plot_lp(lp)


if __name__ == "__main__":
    main()
