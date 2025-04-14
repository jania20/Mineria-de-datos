# ------------------- --------------  PRACTICA 3 -----------------------------------
import matplotlib.pyplot as plt
import seaborn as sns

def practica_3(df=None):
    """Función que contiene TU código original de visualización sin modificaciones"""
    # --- TU CÓDIGO ORIGINAL TAL CUAL ---
    
    # Si no se recibe df, cargamos el dataset limpio
    if df is None:
        try:
            df = pd.read_csv('./cleaned_games_data.csv')
        except FileNotFoundError:
            print("Error: No se encontró 'cleaned_games_data.csv'")
            return
    
    # Configuración general de los gráficos (ACTUALIZADA)
    plt.style.use('seaborn-v0_8')  # o usa 'ggplot', 'fivethirtyeight'
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 6)
    plt.rcParams['font.size'] = 12

    # 1. Histograma de Scores
    plt.figure(figsize=(10, 6))
    df['score'].hist(bins=20, color='skyblue', edgecolor='black', grid=False)
    plt.title('Distribución de Scores de Juegos', pad=20)
    plt.xlabel('Score', labelpad=10)
    plt.ylabel('Frecuencia', labelpad=10)
    plt.savefig('histograma_scores.png', bbox_inches='tight', dpi=300)
    plt.close()

    # 2. Boxplot por Plataforma (top 10)
    plt.figure(figsize=(12, 6))
    top_platforms = df['platform'].value_counts().head(10).index
    df_filtered = df[df['platform'].isin(top_platforms)]
    sns.boxplot(x='platform', y='score', data=df_filtered, palette='viridis')
    plt.title('Distribución de Scores por Plataforma (Top 10)', pad=20)
    plt.xlabel('Plataforma', labelpad=10)
    plt.ylabel('Score', labelpad=10)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('boxplot_plataformas.png', dpi=300)
    plt.close()

    print("\nVisualizaciones generadas y guardadas como imágenes PNG (300 dpi)")
    return df  # Retornamos el DataFrame por si se necesita

# Esto permite ejecutar el script directamente
if __name__ == "__main__":
    import pandas as pd  # Solo necesario para ejecución directa
    practica_3()