"""Parte VI: descenso batch y minibatch SGD."""

from __future__ import annotations

from pathlib import Path
import csv

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs" / "parte_vi"

X = np.array([-1.0, 0.0, 1.0, 2.0], dtype=np.float64)
Y = np.array([-2.0, 0.0, 2.0, 4.0], dtype=np.float64)


def write_csv(path: Path, header: list[str], rows: list[list[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def loss(w: float, x: np.ndarray = X, y: np.ndarray = Y) -> float:
    """Pérdida media cuadrática con la normalización del examen."""
    return float(np.mean((w*x-y)**2))


def gradient(w: float, x: np.ndarray = X, y: np.ndarray = Y) -> float:
    """Gradiente de la pérdida promedio sobre el conjunto recibido."""
    return float(2.0*np.mean(x*(w*x-y)))


def analytical_solution() -> dict[str, float]:
    """VI.1: L(w)=1.5(w-2)^2, L'(w)=3w-6."""
    return {
        "quadratic_a": 1.5,
        "quadratic_b": -6.0,
        "quadratic_c": 6.0,
        "w_star": 2.0,
        "loss_star": 0.0,
    }


def two_batch_updates(alpha: float = 0.1) -> list[dict[str, float]]:
    """VI.2: dos actualizaciones batch desde w0=0."""
    w = 0.0
    rows = [{"k": 0, "w": w, "gradient": gradient(w), "loss": loss(w)}]

    for k in range(1, 3):
        w = w - alpha*gradient(w)
        rows.append({"k": k, "w": w, "gradient": gradient(w), "loss": loss(w)})

    return rows


def batch_gd(epochs: int = 30, alpha: float = 0.1) -> tuple[float, np.ndarray]:
    """VI.3(a): una actualización usando los cuatro datos por época."""
    w = 0.0
    history = [loss(w)]

    for _ in range(epochs):
        w -= alpha*gradient(w)
        history.append(loss(w))

    return w, np.array(history, dtype=np.float64)


def minibatch_sgd(
    seed: int,
    epochs: int = 30,
    alpha: float = 0.1,
    batch_size: int = 2,
) -> tuple[float, np.ndarray]:
    """VI.3(b): permutación nueva por época y dos minibatches de tamaño 2."""
    rng = np.random.default_rng(seed)
    w = 0.0
    history = [loss(w)]

    for _ in range(epochs):
        permutation = rng.permutation(len(X))
        for start in range(0, len(X), batch_size):
            idx = permutation[start:start+batch_size]
            w -= alpha*gradient(w, X[idx], Y[idx])
        history.append(loss(w))

    return w, np.array(history, dtype=np.float64)


def run_comparison() -> tuple[list[list[object]], dict[str, np.ndarray]]:
    """Ejecuta batch y las cinco semillas de minibatch SGD."""
    table: list[list[object]] = []
    histories: dict[str, np.ndarray] = {}

    w_batch, h_batch = batch_gd()
    table.append(["GD_batch", "-", w_batch, loss(w_batch)])
    histories["GD batch"] = h_batch

    for seed in range(5):
        w_sgd, h_sgd = minibatch_sgd(seed=seed)
        table.append(["minibatch_SGD", seed, w_sgd, loss(w_sgd)])
        histories[f"SGD seed {seed}"] = h_sgd

    return table, histories


def plot_histories(histories: dict[str, np.ndarray]) -> Path:
    """VI.3: pérdida completa al final de cada época."""
    fig, ax = plt.subplots(figsize=(9, 6))

    for label, values in histories.items():
        epochs = np.arange(len(values))
        ax.semilogy(epochs, values, label=label)

    ax.set_xlabel("Época")
    ax.set_ylabel("Pérdida completa L(w)")
    ax.set_title("VI.3 — GD batch y minibatch SGD")
    ax.legend()
    ax.grid(True)

    out = OUTPUT / "VI_3_perdida_semilog.png"
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)

    print("\n=== PARTE VI ===")

    ana = analytical_solution()
    print("\nVI.1 — Solución analítica")
    print("L(w) = 1.5 w^2 - 6 w + 6 = 1.5 (w-2)^2")
    print("L'(w) = 3w - 6")
    print(f"w* = {ana['w_star']:.6f}")
    print(f"L(w*) = {ana['loss_star']:.6f}")

    print("\nVI.2 — Dos actualizaciones batch")
    updates = two_batch_updates()
    for row in updates:
        print(
            f"k={row['k']}  w={row['w']:.8f}  "
            f"L'(w)={row['gradient']:.8f}  L(w)={row['loss']:.8f}"
        )

    write_csv(
        OUTPUT / "VI_2_dos_actualizaciones_batch.csv",
        ["k", "w", "gradiente", "perdida"],
        [[r["k"], r["w"], r["gradient"], r["loss"]] for r in updates],
    )

    print("\nVI.3 — Comparación a 30 épocas")
    table, histories = run_comparison()
    for method, seed, w, final_loss in table:
        print(f"{method:15s} seed={str(seed):>2s}  w={w:.12f}  L={final_loss:.6e}")

    write_csv(
        OUTPUT / "VI_3_resultados_finales.csv",
        ["metodo", "semilla", "w_final", "perdida_final"],
        table,
    )

    # Tabla de pérdidas por época para reproducibilidad.
    labels = list(histories.keys())
    rows = []
    n = len(next(iter(histories.values())))
    for epoch in range(n):
        rows.append([epoch] + [histories[label][epoch] for label in labels])

    write_csv(
        OUTPUT / "VI_3_perdida_por_epoca.csv",
        ["epoca"] + labels,
        rows,
    )
    plot_histories(histories)


if __name__ == "__main__":
    main()
