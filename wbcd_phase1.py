#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WBCD Fase I - Carga y Preprocesamiento (un solo archivo)
-------------------------------------------------------
Pasos solicitados:
1) Descargar/Cargar el dataset (usamos el ZIP local entregado por el profesor)
2) Cargar el conjunto de datos en Python (pandas)
3) Preprocesamiento:
   - Sustituir M -> 1, B -> 0 en la etiqueta (diagnosis)
   - Descartar la primera columna (ID)
   - Verificar valores perdidos
Además:
   - Guardar una versión limpia en CSV (wbcd_clean.csv)
   - Mostrar distribución de clases, filas/columnas, y ejemplo de datos
Uso:
  python wbcd_phase1.py --zip "/ruta/al/breast+cancer+wisconsin+diagnostic.zip" --outdir "./salida"
Si no especifica argumentos, intenta usar el ZIP "breast+cancer+wisconsin+diagnostic.zip"
en el directorio actual.
"""

import argparse
from pathlib import Path
from zipfile import ZipFile
import io
import pandas as pd

# Nombres de columnas oficiales de UCI para WDBC (32 columnas)
COLUMN_NAMES = [
    "id", "diagnosis",
    "radius_mean","texture_mean","perimeter_mean","area_mean","smoothness_mean",
    "compactness_mean","concavity_mean","concave_points_mean","symmetry_mean","fractal_dimension_mean",
    "radius_se","texture_se","perimeter_se","area_se","smoothness_se",
    "compactness_se","concavity_se","concave_points_se","symmetry_se","fractal_dimension_se",
    "radius_worst","texture_worst","perimeter_worst","area_worst","smoothness_worst",
    "compactness_worst","concavity_worst","concave_points_worst","symmetry_worst","fractal_dimension_worst"
]

def find_data_member(z: ZipFile):
    """
    Localiza dentro del ZIP un archivo de datos válido.
    Preferencias:
      - 'wdbc.data' (formato UCI, sin cabecera)
      - archivo .csv
      - archivo .data
    Devuelve (member_name, has_header)
    """
    names = z.namelist()
    # Limpieza de nombres (evitar carpetas)
    files = [n for n in names if not n.endswith("/")]
    # Prioridad 1: wdbc.data (clásico de UCI)
    for nm in files:
        if nm.lower().endswith("wdbc.data"):
            return nm, False
    # Prioridad 2: algún .csv
    for nm in files:
        if nm.lower().endswith(".csv"):
            return nm, True  # normalmente .csv trae cabecera
    # Prioridad 3: algún .data genérico
    for nm in files:
        if nm.lower().endswith(".data"):
            return nm, False
    raise FileNotFoundError("No se encontró un archivo de datos (.csv o .data) dentro del ZIP.")

def load_wbcd_from_zip(zip_path: Path) -> pd.DataFrame:
    """
    Carga el dataset desde un ZIP con formato UCI o CSV.
    Estandariza columnas y asegura que exista 'diagnosis' como M/B.
    """
    if not zip_path.exists():
        raise FileNotFoundError(f"No existe el ZIP: {zip_path}")
    with ZipFile(zip_path, "r") as z:
        member, has_header = find_data_member(z)
        with z.open(member) as f:
            raw = f.read()
    buf = io.BytesIO(raw)

    if has_header:
        df = pd.read_csv(buf)
    else:
        # sin cabecera: asignamos nombres oficiales
        df = pd.read_csv(buf, header=None, names=COLUMN_NAMES)

    # Normalizamos nombres de columnas a minúsculas y con guiones bajos
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Garantizamos que exista la columna diagnosis
    if "diagnosis" not in df.columns:
        raise ValueError("No se encontró la columna 'diagnosis' en el archivo.")

    # Si la primera columna es ID o similar, la renombramos a 'id' para luego descartarla.
    first_col = df.columns[0]
    if first_col != "id":
        # si detectamos algo que luce como ID numérico, lo renombramos
        # criterio simple: muchos valores únicos y numéricos
        try:
            if pd.api.types.is_numeric_dtype(df[first_col]) and df[first_col].nunique() > (0.9 * len(df)):
                df = df.rename(columns={first_col: "id"})
        except Exception:
            pass

    return df

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    - Mapear diagnosis: M->1, B->0
    - Eliminar primera columna (ID)
    - Limpiar y verificar faltantes
    """
    # Mapear diagnosis
    mapping = {"M": 1, "B": 0, "m": 1, "b": 0}
    if pd.api.types.is_numeric_dtype(df["diagnosis"]):
        # Ya podría venir en 0/1; solo aseguramos tipo int
        df["diagnosis"] = df["diagnosis"].astype(int)
    else:
        df["diagnosis"] = df["diagnosis"].map(mapping)
        if df["diagnosis"].isna().any():
            # Intento de corrección si viene como strings "Maligno"/"Benigno"
            alt_map = {"Maligno": 1, "Benigno": 0, "maligno": 1, "benigno": 0}
            df["diagnosis"] = df["diagnosis"].fillna(df["diagnosis"].replace(alt_map))
        # Al final, convertir a int (si queda algo raro, fallará claramente)
        df["diagnosis"] = df["diagnosis"].astype(int)

    # Descartar la primera columna (ID) si existe
    # Aseguramos el orden original: 'id' debería ser la primera si venía el formato UCI clásico.
    cols = list(df.columns)
    if cols[0] == "id":
        df = df.drop(columns=["id"])
    else:
        # por si acaso: eliminar cualquier columna que claramente sea ID
        candidate_ids = [c for c in df.columns if c in ("id", "unnamed: 0")]
        if candidate_ids:
            df = df.drop(columns=candidate_ids)

    # Reemplazar símbolos de faltantes comunes con NaN y convertir numéricos
    df = df.replace({"?": pd.NA, "NA": pd.NA, "null": pd.NA, "None": pd.NA})
    # Convertir todas menos 'diagnosis' a numericas si es posible
    feature_cols = [c for c in df.columns if c != "diagnosis"]
    for c in feature_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df

def report(df: pd.DataFrame):
    """
    Imprime resumenes: forma, faltantes, distribución de clases y ejemplo de filas.
    """
    print("=== PASO 2: Carga del dataset ===")
    print(f"Filas x Columnas: {df.shape[0]} x {df.shape[1]}")
    print()

    print("=== PASO 3: Verificación de valores perdidos ===")
    missing = df.isna().sum()
    total_missing = int(missing.sum())
    if total_missing == 0:
        print("No se encontraron valores perdidos.")
    else:
        print("Valores perdidos por columna:")
        print(missing[missing > 0].sort_values(ascending=False))
    print()

    print("=== Distribución de la variable objetivo (diagnosis: 1=maligno, 0=benigno) ===")
    counts = df["diagnosis"].value_counts().sort_index()
    print(counts.to_string())
    print()

    print("=== Vista previa de datos (5 filas) ===")
    print(df.head(5).to_string())
    print()

def main():
    parser = argparse.ArgumentParser(description="WBCD Fase I - Carga y Preprocesamiento")
    parser.add_argument("--zip", dest="zip_path", type=str, default="breast+cancer+wisconsin+diagnostic.zip",
                        help="Ruta al ZIP con el dataset (por defecto: ./breast+cancer+wisconsin+diagnostic.zip)")
    parser.add_argument("--outdir", dest="outdir", type=str, default=".",
                        help="Carpeta de salida para wbcd_clean.csv (por defecto: .)")
    args = parser.parse_args()

    zip_path = Path(args.zip_path)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("=== PASO 1: Carga desde ZIP ===")
    print(f"Usando ZIP: {zip_path.resolve()}")
    df_raw = load_wbcd_from_zip(zip_path)
    print("Datos cargados desde el ZIP.")

    print("\n=== PASO 2: Normalización de columnas y tipos ===")
    df_clean = preprocess(df_raw)
    print("Preprocesamiento aplicado: diagnosis mapeado (M/B -> 1/0) y columna ID eliminada (si existía).")

    print("\n=== PASO 3: Reporte rápido ===")
    report(df_clean)

    # Guardar CSV limpio
    out_csv = outdir / "wbcd_clean.csv"
    df_clean.to_csv(out_csv, index=False)
    print(f"Archivo limpio guardado en: {out_csv.resolve()}")

if __name__ == "__main__":
    main()
