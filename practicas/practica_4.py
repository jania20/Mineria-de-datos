# PRÁCTICA 4: PRUEBAS ESTADÍSTICAS
# Objetivo: Comparar scores entre géneros y plataformas usando ANOVA, Kruskal-Wallis y Tukey HSD

import pandas as pd
import numpy as np
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import matplotlib.pyplot as plt
import seaborn as sns

def practica_4(df):
    """Función principal que realiza el análisis estadístico"""
    # Configuración de gráficos
    plt.rcParams['figure.figsize'] = (12, 6)
    plt.rcParams['font.size'] = 12
    sns.set_theme(style="whitegrid")
    
    # 1. PREPARAR DATOS (usando el dataframe recibido como parámetro)
    df = df.copy()  # Trabajamos sobre una copia para no modificar el original
    df['r-date'] = pd.to_datetime(df['r-date'], errors='coerce')
    df['year'] = df['r-date'].dt.year

    # Filtrar géneros y plataformas con suficientes datos
    top_genres = df['genre'].value_counts().head(5).index.tolist()
    top_platforms = df['platform'].value_counts().head(5).index.tolist()

    df_filtered = df[
        (df['genre'].isin(top_genres)) & 
        (df['platform'].isin(top_platforms))
    ].copy()

    print("\n=== PRÁCTICA 4: ANÁLISIS ESTADÍSTICO ===")
    print("📊 Datos filtrados para análisis:")
    print(f"- Géneros analizados: {top_genres}")
    print(f"- Plataformas analizadas: {top_platforms}")
    print(f"- Filas en dataset: {len(df_filtered)}")

    # 2. ANOVA: ¿LOS SCORES VARÍAN POR GÉNERO?
    print("\n🔍 ANOVA: ¿Diferencias significativas entre géneros?")

    model = ols('score ~ C(genre)', data=df_filtered).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    print("\nTabla ANOVA:")
    print(anova_table)

    alpha = 0.05
    p_value = anova_table['PR(>F)'][0]
    if p_value < alpha:
        print(f"\n✅ CONCLUSIÓN (p={p_value:.4f} < α={alpha}): Hay diferencias significativas entre géneros.")
    else:
        print(f"\n❌ CONCLUSIÓN (p={p_value:.4f} > α={alpha}): No hay diferencias significativas.")

    # Gráfico de cajas por género
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='genre', y='score', data=df_filtered, palette='viridis')
    plt.title('Distribución de Scores por Género', fontsize=14)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('anova_scores_genero.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("\n📌 Gráfico guardado: 'anova_scores_genero.png'")

    # 3. PRUEBA POST-HOC (TUKEY HSD)
    if p_value < alpha:
        print("\n🔍 PRUEBA TUKEY HSD: ¿Qué géneros difieren?")
        tukey = pairwise_tukeyhsd(
            endog=df_filtered['score'],
            groups=df_filtered['genre'],
            alpha=alpha
        )
        print(tukey.summary())

        tukey.plot_simultaneous()
        plt.title('Comparación de Géneros (Tukey HSD)')
        plt.savefig('tukey_genres.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("\n📌 Gráfico guardado: 'tukey_genres.png'")

    # 4. KRUSKAL-WALLIS: ¿LOS SCORES VARÍAN POR PLATAFORMA?
    print("\n🔍 KRUSKAL-WALLIS: ¿Diferencias entre plataformas?")

    platform_groups = [group['score'].values for name, group in df_filtered.groupby('platform')]

    h_stat, p_value = stats.kruskal(*platform_groups)
    print(f"\nEstadístico H: {h_stat:.2f}, p-valor: {p_value:.4f}")

    if p_value < alpha:
        print(f"\n✅ CONCLUSIÓN (p={p_value:.4f} < α={alpha}): Hay diferencias significativas entre plataformas.")
    else:
        print(f"\n❌ CONCLUSIÓN (p={p_value:.4f} > α={alpha}): No hay diferencias significativas.")

    # Gráfico de cajas por plataforma
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='platform', y='score', data=df_filtered, palette='magma')
    plt.title('Distribución de Scores por Plataforma', fontsize=14)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('kruskal_scores_plataforma.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("\n📌 Gráfico guardado: 'kruskal_scores_plataforma.png'")

    print("\n🎉 ¡Análisis completado! Verifica los gráficos guardados.")
    
    return df  # Devolvemos el dataframe (opcional, según necesidades)