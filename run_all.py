"""Ejecuta las Partes IV, V y VI y genera todos los resultados."""

from src.parte_iv import main as run_iv
from src.parte_v import main as run_v
from src.parte_vi import main as run_vi


def main() -> None:
    run_iv()
    run_v()
    run_vi()
    print("\nProyecto ejecutado correctamente.")
    print("Revise la carpeta outputs/ para las tablas y gráficas.")


if __name__ == "__main__":
    main()
