import pandas as pd

 #Cargar datos
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

 #Guardar dataset limpio
df.to_csv('./cleaned_games_data.csv', index=False)
print("\nDataset limpio guardado como 'cleaned_games_data.csv'")


#practica 2
import pandas as pd

# Cargar datos
df = pd.read_csv('./cleaned_games_data.csv')

# Configuración para mostrar mejor los datos en consola
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 20)

# Limpieza adicional de datos - FORMA CORREGIDA
df = df.replace('', pd.NA)  # Convertir strings vacíos a NA
df = df.dropna(subset=['score'])  # Eliminar filas sin score

print("\n=== PRIMERAS 5 FILAS ===")
print(df.head(5).to_string(index=False))

print("\n\n=== PRIMERAS 10 FILAS ===")
print(df.head(10).to_string(index=False))

print("\n\n=== VERIFICACIÓN DE REQUISITOS ===")
print(f"Total de filas: {len(df)}")
print("\nTipos de datos por columna:")
print(df.dtypes.to_string())

print("\nColumnas disponibles:")
print(df.columns.to_list())

print("\n=== ESTADÍSTICAS NUMÉRICAS BÁSICAS ===")
print(df.describe().to_string())

print("\n=== ESTADÍSTICAS POR GÉNERO ===")
print(df.groupby('genre')['score'].describe())

print("\n=== ESTADÍSTICAS POR PLATAFORMA ===")
print(df.groupby('platform')['score'].agg(['mean', 'count', 'std']).sort_values('mean', ascending=False))

# CORRELACIONES - VERSIÓN CORREGIDA
print("\n=== CORRELACIONES ENTRE VARIABLES NUMÉRICAS ===")
numeric_df = df[['score', 'user score', 'players']].apply(pd.to_numeric, errors='coerce').dropna()
print(numeric_df.corr())

print("""
=== DIAGRAMA DE RELACIONES COMPLETO ===

ENTIDAD PRINCIPAL: JUEGO
- Atributos: name (PK), score, user_score, r-date, players
- Relaciones:
  * Pertenece a 1 GÉNERO (relación 1-a-muchos)
  * Disponible en 1 PLATAFORMA (relación 1-a-muchos)
  * Publicado por 1 EDITOR (relación 1-a-muchos si existe columna publisher)

ENTIDAD: GÉNERO
- Atributos: genre (PK)
- Relaciones: Contiene muchos JUEGOS

ENTIDAD: PLATAFORMA
- Atributos: platform (PK)
- Relaciones: Aloja muchos JUEGOS

ENTIDAD: EDITOR (si existe en los datos)
- Atributos: publisher (PK)
- Relaciones: Publica muchos JUEGOS
""")

# Guardar resultados en un archivo de texto
with open('resultados_analisis.txt', 'w', encoding='utf-8') as f:
    f.write("=== RESULTADOS DEL ANÁLISIS ===\n\n")
    f.write("Primeras 5 filas:\n")
    f.write(df.head(5).to_string(index=False) + "\n\n")
    f.write("Estadísticas descriptivas:\n")
    f.write(df.describe().to_string() + "\n\n")
    f.write("\n=== ESTADÍSTICAS POR GÉNERO ===\n")
    f.write(df.groupby('genre')['score'].describe().to_string() + "\n\n")
    f.write("\n=== ESTADÍSTICAS POR PLATAFORMA ===\n")
    f.write(df.groupby('platform')['score'].agg(['mean', 'count', 'std']).sort_values('mean', ascending=False).to_string() + "\n\n")
    f.write("\n=== CORRELACIONES ===\n")
    f.write(numeric_df.corr().to_string() + "\n\n")
    f.write("Tipos de datos:\n")
    f.write(df.dtypes.to_string() + "\n")

print("\nAnálisis completado. Resultados guardados en 'resultados_analisis.txt'")

# Estadísticas por año
if 'r-date' in df.columns:
    df['year'] = df['r-date'].dt.year
    print("\n=== ESTADÍSTICAS POR AÑO ===")
    print(df.groupby('year')['score'].agg(['mean', 'count']).sort_values('mean', ascending=False))

# Estadísticas por publisher (si existe)
if 'publisher' in df.columns:
    print("\n=== MEJORES EDITORES (SCORE PROMEDIO) ===")
    print(df.groupby('publisher')['score'].agg(['mean', 'count']).sort_values('mean', ascending=False).head(10))