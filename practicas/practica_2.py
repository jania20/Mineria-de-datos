

# ------------------------- PRACTICA 2 -------------------------------------------
import pandas as pd

def practica_2(df=None):
    """Versión modularizada de tu práctica 2 original (sin cambios en la lógica)"""
    # --- Código IDÉNTICO al tuyo, solo con indentación correcta ---
    if df is None:
        df = pd.read_csv('./cleaned_games_data.csv')
    
    df['r-date'] = pd.to_datetime(df['r-date'], errors='coerce')
    df = df.dropna(subset=['r-date', 'score'])
    
    # Configuración de visualización (tu código original)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.max_colwidth', 20)
    df = df.replace('', pd.NA)
    
    # Todos tus print() originales
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
    
    # Correlaciones (tu código original)
    numeric_df = df[['score', 'user score', 'players']].apply(pd.to_numeric, errors='coerce').dropna()
    print("\n=== CORRELACIONES ENTRE VARIABLES NUMÉRICAS ===")
    print(numeric_df.corr())
    
    # Diagrama de relaciones (texto idéntico al tuyo)
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
    
    # Guardado de resultados (tu código original)
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

    # Estadísticas por año (tu código original)
    if 'r-date' in df.columns:
        df['year'] = df['r-date'].dt.year
        print("\n=== ESTADÍSTICAS POR AÑO ===")
        print(df.groupby('year')['score'].agg(['mean', 'count']).sort_values('mean', ascending=False))

    # Estadísticas por editor (tu código original)
    if 'publisher' in df.columns:
        print("\n=== MEJORES EDITORES (SCORE PROMEDIO) ===")
        print(df.groupby('publisher')['score'].agg(['mean', 'count']).sort_values('mean', ascending=False).head(10))
    
    return df  # Retorna el DataFrame para usar en siguientes prácticas

# Bloque para ejecución independiente (opcional)
if __name__ == "__main__":
    practica_2()