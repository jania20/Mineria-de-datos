import pandas as pd

# Cargar datos
df = pd.read_csv('./archive/games-data.csv')

# Limpieza inicial
df.columns = df.columns.str.strip()

# Convertir fechas
df['r-date'] = pd.to_datetime(df['r-date'], format='%B %d, %Y', errors='coerce')

# Limpiar columnas numéricas
numeric_cols = ['score', 'user score', 'players', 'critics', 'users']
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Eliminar filas con datos críticos faltantes
df = df.dropna(subset=['name', 'r-date', 'score'])

# Mostrar las primeras filas
print("=== Primeras 5 filas del dataset ===")
print(df.head())

print("\n=== Primeras 10 filas del dataset ===")
print(df.head(10))

# Verificación de requisitos
print("\n=== Verificación de Requisitos ===")
print(f"Filas totales: {len(df)}")
print("\nTipos de datos:")
print(df.dtypes)

# Guardar dataset limpio
df.to_csv('./cleaned_games_data.csv', index=False)
print("\nDataset limpio guardado como 'cleaned_games_data.csv'")