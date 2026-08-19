"""
Taller: Preprocesamiento de Datos para Clasificación con Redes Neuronales
Dataset: Pacientes Cardíacos
Autor: Santiago
"""

import pandas as pd
import numpy as np
import json

# ──────────────────────────────────────────────────────────────
# 1. CARGA DEL DATASET
# ──────────────────────────────────────────────────────────────
URL = "https://raw.githubusercontent.com/adiacla/bigdata/refs/heads/master/pacientes.csv"

df = pd.read_csv(URL)
print("Dataset original:")
print(f"  Filas: {df.shape[0]}, Columnas: {df.shape[1]}")
print(f"  Columnas: {df.columns.tolist()}")
print(f"  Valores nulos:\n{df.isnull().sum()}\n")

# ──────────────────────────────────────────────────────────────
# 2. SELECCIÓN Y RENOMBRAMIENTO DE COLUMNAS
#    edad       → x      (Feature 1)
#    colesterol → y      (Feature 2)
#    problema_cardiaco → label (Target)
# ──────────────────────────────────────────────────────────────
df = df[['edad', 'colesterol', 'problema_cardiaco']].copy()
df.columns = ['x', 'y', 'label']

# ──────────────────────────────────────────────────────────────
# 3. LIMPIEZA: eliminar filas con valores nulos
# ──────────────────────────────────────────────────────────────
df = df.dropna().reset_index(drop=True)
print(f"Registros válidos tras limpiar nulos: {len(df)}")

# ──────────────────────────────────────────────────────────────
# 4. MAPEO DE ETIQUETAS
#    1 (con problema cardíaco)  → label =  1  (clase positiva)
#    0 (sin problema cardíaco)  → label = -1  (clase negativa)
# ──────────────────────────────────────────────────────────────
df['label'] = df['label'].map({1: 1, 0: -1}).astype(int)

print(f"\nDistribución de clases:")
print(f"  label =  1 (positivo): {(df['label'] == 1).sum()}")
print(f"  label = -1 (negativo): {(df['label'] == -1).sum()}")

# ──────────────────────────────────────────────────────────────
# 5. ESTANDARIZACIÓN Z-SCORE × 2
#    x_norm = ((x - μ) / σ) × 2
# ──────────────────────────────────────────────────────────────
for col in ['x', 'y']:
    mu  = df[col].mean()
    std = df[col].std()
    df[col] = ((df[col] - mu) / std) * 2
    df[col] = df[col].round(4)
    print(f"\nEstandarización '{col}':  μ={mu:.2f}, σ={std:.2f}")

print(f"\nEstadísticas post-normalización:")
print(df[['x', 'y']].describe().round(4))

# ──────────────────────────────────────────────────────────────
# 6. EXPORTAR A JSON
# ──────────────────────────────────────────────────────────────
records = df.to_dict(orient='records')
# Ruta relativa al script (funciona desde cualquier directorio)
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.join(script_dir, '..', 'data', 'pacientes_nn.json')

with open(output_path, 'w') as f:
    json.dump(records, f, indent=2)

print(f"\n✅ JSON exportado: {output_path}")
print(f"   Total registros: {len(records)}")
print(f"\nMuestra (primeros 3 registros):")
print(json.dumps(records[:3], indent=2))
