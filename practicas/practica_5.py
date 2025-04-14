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
    Versión corregida que usa solo columnas disponibles
    """
    print("\n=== PRÁCTICA 5: MODELOS LINEALES Y CORRELACIÓN ===")
    
    # 1. Preparación de datos
    df = df.copy()
    
    # Verificar columnas disponibles
    print("Columnas disponibles:", df.columns.tolist())
    
    # Codificar variables categóricas
    le = LabelEncoder()
    df_encoded = df.copy()
    categorical_cols = ['genre', 'platform', 'developer', 'rating']
    for col in categorical_cols:
        if col in df.columns:
            df_encoded[col] = le.fit_transform(df[col].astype(str))
    
    # 2. Análisis de correlación con columnas numéricas disponibles
    numeric_cols = df_encoded.select_dtypes(include=[np.number]).columns.tolist()
    print("\n🔍 MATRIZ DE CORRELACIÓN (columnas numéricas)")
    print("Columnas usadas:", numeric_cols)
    
    plt.figure(figsize=(12, 8))
    corr_matrix = df_encoded[numeric_cols].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Matriz de Correlación', fontsize=14)
    plt.tight_layout()
    plt.savefig('correlation_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("📌 Gráfico guardado: 'correlation_matrix.png'")
    
    # 3. Modelo de regresión solo si hay suficientes features
    if 'score' in numeric_cols and len(numeric_cols) > 1:
        # Usar todas las columnas numéricas excepto score como features
        features = [col for col in numeric_cols if col != 'score']
        X = df_encoded[features]
        y = df_encoded['score']
        
        # Manejo de valores faltantes
        X = X.replace([np.inf, -np.inf], np.nan).dropna()
        y = y[X.index]
        
        if len(X) > 0:
            print("\n📈 MODELO DE REGRESIÓN LINEAL")
            print("Variables independientes:", features)
            print("Variable dependiente: score")
            
            model = LinearRegression()
            model.fit(X, y)
            
            y_pred = model.predict(X)
            r2 = r2_score(y, y_pred)
            
            print("\nCoeficientes del modelo:")
            for feature, coef in zip(features, model.coef_):
                print(f"{feature}: {coef:.4f}")
            print(f"Intercepto: {model.intercept_:.4f}")
            print(f"\nR² score: {r2:.4f}")
            
            # Visualización
            plt.figure(figsize=(10, 6))
            sns.regplot(x=y_pred, y=y, line_kws={'color': 'red'})
            plt.xlabel('Predicciones del modelo')
            plt.ylabel('Valores reales')
            plt.title(f'Regresión Lineal (R² = {r2:.2f})', fontsize=14)
            plt.tight_layout()
            plt.savefig('linear_regression.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("📌 Gráfico guardado: 'linear_regression.png'")
            
            # Gráfico de residuos
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
        else:
            print("\n⚠️ No hay suficientes datos después de limpieza para el modelo")
    else:
        print("\n⚠️ No hay suficientes variables numéricas para regresión")
    
    print("\n🎉 ¡Análisis completado! Verifica los gráficos guardados.")
    return df