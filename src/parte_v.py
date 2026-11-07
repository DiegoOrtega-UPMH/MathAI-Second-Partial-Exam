"""Parte V: análisis numérico, condicionamiento y regularización."""

from __future__ import annotations

from pathlib import Path
import csv
import math

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs" / "parte_v"


def write_csv(path: Path, header: list[str], rows: list[list[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def g(x: np.ndarray | float) -> np.ndarray | float:
    return np.log1p(x)


def forward_difference(h: float) -> float:
    return float((g(1.0+h)-g(1.0))/h)


def central_difference(h: float) -> float:
    return float((g(1.0+h)-g(1.0-h))/(2.0*h))


def finite_difference_table() -> list[dict[str, float]]:
    """V.A.2: tabla float64 y orden empírico."""
    exact = 0.5
    rows: list[dict[str, float]] = []

    previous_e_plus = None
    previous_e_central = None

    for k in range(1, 9):
        h = float(10.0**(-k))
        d_plus = forward_difference(h)
        d_central = central_difference(h)
        e_plus = abs(d_plus-exact)
        e_central = abs(d_central-exact)

        p_plus = (
            math.log(previous_e_plus/e_plus)/math.log(10.0)
            if previous_e_plus is not None and e_plus > 0
            else float("nan")
        )
        p_central = (
            math.log(previous_e_central/e_central)/math.log(10.0)
            if previous_e_central is not None and e_central > 0
            else float("nan")
        )

        rows.append(
            {
                "k": k,
                "h": h,
                "D_plus": d_plus,
                "D_central": d_central,
                "E_plus": e_plus,
                "E_central": e_central,
                "p_plus": p_plus,
                "p_central": p_central,
            }
        )
        previous_e_plus = e_plus
        previous_e_central = e_central

    return rows


def plot_finite_difference_errors(rows: list[dict[str, float]]) -> Path:
    """V.A.3: errores y referencias de orden 1 y 2."""
    h = np.array([r["h"] for r in rows])
    e_plus = np.array([r["E_plus"] for r in rows])
    e_central = np.array([r["E_central"] for r in rows])

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.loglog(h, e_plus, label="Error D_h^+")
    ax.loglog(h, e_central, label="Error D_h^0")
    ax.loglog(h, h/8.0, label="Referencia proporcional a h")
    ax.loglog(h, h**2/24.0, label="Referencia proporcional a h²")
    ax.set_xlabel("h")
    ax.set_ylabel("Error absoluto")
    ax.set_title("V.A — Truncamiento, redondeo y orden de convergencia")
    ax.legend()
    ax.grid(True)

    out = OUTPUT / "V_A_errores_loglog.png"
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def symbolic_conditioning() -> dict[str, sp.Expr]:
    """V.B.1: expresiones simbólicas del sistema casi singular."""
    eps, delta = sp.symbols("epsilon delta", positive=True)
    A = sp.Matrix([[1, 1], [1, 1+eps]])
    b = sp.Matrix([2, 2+eps])
    bd = sp.Matrix([2, 2+eps+delta])

    x = sp.simplify(A.inv()*b)
    xd = sp.simplify(A.inv()*bd)

    trace = sp.trace(A)
    determinant = A.det()
    disc = sp.simplify(trace**2 - 4*determinant)
    lam_plus = sp.simplify((trace + sp.sqrt(disc))/2)
    lam_minus = sp.simplify((trace - sp.sqrt(disc))/2)

    return {
        "x1": x[0],
        "x2": x[1],
        "xd1": xd[0],
        "xd2": xd[1],
        "lambda_plus": lam_plus,
        "lambda_minus": lam_minus,
    }


def conditioning_numeric(eps: float = 1e-8, delta: float = 1e-6, lam: float = 0.01) -> dict[str, object]:
    """V.B.2 y V.B.3: comparación sin y con regularización."""
    A = np.array([[1.0, 1.0], [1.0, 1.0+eps]], dtype=np.float64)
    b = np.array([2.0, 2.0+eps], dtype=np.float64)
    db = np.array([0.0, delta], dtype=np.float64)

    x = np.linalg.solve(A, b)
    x_delta = np.linalg.solve(A, b+db)
    cond_A = float(np.linalg.cond(A, 2))

    rel_b = float(np.linalg.norm(db)/np.linalg.norm(b))
    rel_x = float(np.linalg.norm(x_delta-x)/np.linalg.norm(x))
    amplification = rel_x/rel_b

    A_ridge = A + lam*np.eye(2)
    x_ridge = np.linalg.solve(A_ridge, b)
    x_ridge_delta = np.linalg.solve(A_ridge, b+db)
    cond_ridge = float(np.linalg.cond(A_ridge, 2))
    rel_x_ridge = float(np.linalg.norm(x_ridge_delta-x_ridge)/np.linalg.norm(x_ridge))
    amplification_ridge = rel_x_ridge/rel_b

    return {
        "A": A,
        "b": b,
        "db": db,
        "x": x,
        "x_delta": x_delta,
        "cond_A": cond_A,
        "norm_b": float(np.linalg.norm(b)),
        "rel_b": rel_b,
        "rel_x": rel_x,
        "amplification": amplification,
        "lambda": lam,
        "A_ridge": A_ridge,
        "x_ridge": x_ridge,
        "x_ridge_delta": x_ridge_delta,
        "cond_ridge": cond_ridge,
        "rel_x_ridge": rel_x_ridge,
        "amplification_ridge": amplification_ridge,
        "ridge_bias_norm": float(np.linalg.norm(x_ridge-x)),
    }


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)

    print("\n=== PARTE V ===")

    print("\nV.A — Diferencias finitas")
    rows = finite_difference_table()
    for r in rows:
        print(
            f"h={r['h']:.0e}  "
            f"D+={r['D_plus']:.12f}  "
            f"D0={r['D_central']:.12f}  "
            f"E+={r['E_plus']:.3e}  "
            f"E0={r['E_central']:.3e}  "
            f"p+={r['p_plus']:.4f}  "
            f"p0={r['p_central']:.4f}"
        )

    write_csv(
        OUTPUT / "V_A_diferencias_finitas.csv",
        ["k", "h", "D_plus", "D_central", "E_plus", "E_central", "p_plus", "p_central"],
        [
            [
                r["k"], r["h"], r["D_plus"], r["D_central"],
                r["E_plus"], r["E_central"], r["p_plus"], r["p_central"]
            ]
            for r in rows
        ],
    )
    plot_finite_difference_errors(rows)

    sym = symbolic_conditioning()
    print("\nV.B.1 — Solución simbólica")
    print(f"x = ({sym['x1']}, {sym['x2']})")
    print(f"x_delta = ({sym['xd1']}, {sym['xd2']})")
    print(f"lambda_max = {sym['lambda_plus']}")
    print(f"lambda_min = {sym['lambda_minus']}")

    num = conditioning_numeric()
    print("\nV.B.2 — Sistema sin regularización")
    print(f"cond_2(A) = {num['cond_A']:.6e}")
    print(f"x = {num['x']}")
    print(f"x_delta = {num['x_delta']}")
    print(f"||b||_2 = {num['norm_b']:.12f}")
    print(f"cambio relativo en b = {num['rel_b']:.6e}")
    print(f"cambio relativo en x = {num['rel_x']:.6e}")
    print(f"factor de amplificación observado = {num['amplification']:.6e}")

    print("\nV.B.3 — Regularización lambda=0.01")
    print(f"cond_2(A + lambda I) = {num['cond_ridge']:.6e}")
    print(f"x_R = {num['x_ridge']}")
    print(f"x_R_delta = {num['x_ridge_delta']}")
    print(f"cambio relativo regularizado = {num['rel_x_ridge']:.6e}")
    print(f"factor de amplificación regularizado = {num['amplification_ridge']:.6e}")
    print(f"norma del sesgo respecto a x = {num['ridge_bias_norm']:.6e}")

    write_csv(
        OUTPUT / "V_B_condicionamiento_ridge.csv",
        ["metodo", "condicion_2", "x1", "x2", "x1_perturbado", "x2_perturbado", "cambio_relativo"],
        [
            [
                "sin_regularizacion",
                num["cond_A"],
                num["x"][0],
                num["x"][1],
                num["x_delta"][0],
                num["x_delta"][1],
                num["rel_x"],
            ],
            [
                "ridge_lambda_0.01",
                num["cond_ridge"],
                num["x_ridge"][0],
                num["x_ridge"][1],
                num["x_ridge_delta"][0],
                num["x_ridge_delta"][1],
                num["rel_x_ridge"],
            ],
        ],
    )


if __name__ == "__main__":
    main()
