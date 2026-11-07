# Examen 2 — Matemáticas para Inteligencia Artificial

Proyecto en Python correspondiente a las **Partes IV, V y VI** del examen.

El proyecto conserva la numeración original de los reactivos y separa claramente:

- desarrollo analítico;
- verificación computacional;
- tablas solicitadas;
- gráficas;
- interpretación de resultados.

## Estructura

```text
Examen2_Partes_IV_V_VI/
├── README.md
├── Examen2_Partes_IV_V_VI.ipynb
├── SOLUCIONES.md
├── requirements.txt
├── run_all.py
├── Examen2_Partes_IV_V_VI.code-workspace
├── src/
│   ├── __init__.py
│   ├── parte_iv.py
│   ├── parte_v.py
│   └── parte_vi.py
├── tests/
│   └── test_solutions.py
└── outputs/
    ├── parte_iv/
    ├── parte_v/
    └── parte_vi/
```

## Notebook principal

El archivo `Examen2_Partes_IV_V_VI.ipynb` contiene el desarrollo completo de las Partes IV, V y VI con explicación, código, tablas y gráficas en un solo flujo.

## Preparación en VS Code

1. Abrir la carpeta del proyecto o el archivo `Examen2_Partes_IV_V_VI.code-workspace`.
2. Crear un entorno virtual:

### Windows PowerShell

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

3. Ejecutar todo:

```bash
python run_all.py
```

4. Ejecutar las pruebas:

```bash
python -m unittest discover -s tests -v
```

## Ejecución por parte

```bash
python -m src.parte_iv
python -m src.parte_v
python -m src.parte_vi
```

Cada script imprime los resultados principales y escribe las tablas y figuras dentro de `outputs/`.

## Dependencias

- NumPy: cálculo numérico.
- SymPy: verificación simbólica.
- Matplotlib: gráficas.

No se utiliza `scikit-learn` ni un optimizador de alto nivel para el descenso de gradiente: los algoritmos solicitados se implementan directamente.
