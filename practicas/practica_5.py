import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder
from scipy import stats

def practica_5(df):
    """
    Realiza análisis de correlación y modelos lineales:
    1. Análisis de correlación entre variables numéricas
    2. Modelo de regresión lineal para predecir score
    3. Visualización de resultados
    """
    print("\n=== PRÁCTICA 5: MODELOS LINEALES Y CORRELACIÓN ===")
    
    # 1. Preparación de datos
    df = df.copy()
    
    # Codificar variables categóricas para análisis de correlación
    le = LabelEncoder()
    df_encoded = df.copy()
    categorical_cols = ['genre', 'platform', 'developer', 'rating']
    for col in categorical_cols:
        if col in df.columns:
            df_encoded[col] = le.fit_transform(df[col].astype(str))
    
    # Análisis de correlación
    print("\n🔍 MATRIZ DE CORRELACIÓN")
    plt.figure(figsize=(12, 8))
    corr_matrix = df_encoded.select_dtypes(include=[np.number]).corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Matriz de Correlación', fontsize=14)
    plt.tight_layout()
    plt.savefig('correlation_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("📌 Gráfico guardado: 'correlation_matrix.png'")
    
    # Preparación para modelo lineal
    X = df_encoded[['year', 'plays', 'playing', 'backlogs', 'wishlist']]
    y = df_encoded['score']
    
    # Filtrar valores infinitos y NaN
    X = X.replace([np.inf, -np.inf], np.nan).dropna()
    y = y[X.index]
    
    # Modelo de regresión lineal
    print("\n📈 MODELO DE REGRESIÓN LINEAL")
    model = LinearRegression()
    model.fit(X, y)
    
    # Predicciones
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    
    print(f"\nCoeficientes del modelo:")
    for feature, coef in zip(X.columns, model.coef_):
        print(f"{feature}: {coef:.4f}")
    print(f"Intercepto: {model.intercept_:.4f}")
    print(f"\nR² score: {r2:.4f}")
    
    # Visualización de resultados
    print("\n📊 VISUALIZACIÓN DE RESULTADOS")
    plt.figure(figsize=(10, 6))
    sns.regplot(x=y_pred, y=y, line_kws={'color': 'red'})
    plt.xlabel('Predicciones del modelo')
    plt.ylabel('Valores reales')
    plt.title(f'Regresión Lineal (R² = {r2:.2f})', fontsize=14)
    plt.tight_layout()
    plt.savefig('linear_regression.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("📌 Gráfico guardado: 'linear_regression.png'")
    
    # 6. Gráfico de residuos
    residuals = y - y_pred
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_pred, y=residuals)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicciones del modelo')
    plt.ylabel('Residuos')
    plt.title('Análisis de Residuos', fontsize=14)
    plt.tight_layout()
    plt.savefig('residuals_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("📌 Gráfico guardado: 'residuals_plot.png'")
    
    print("\n🎉 ¡Análisis completado! Verifica los gráficos guardados.")
    return df