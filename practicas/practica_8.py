import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

def practica_8(df):
    """
    Predicción de datos futuros usando regresión lineal en una serie temporal.
    Asume que el DataFrame tiene una columna 'year' (o similar) para la variable temporal.
    """
    print("\n=== PRÁCTICA 8: FORECASTING CON REGRESIÓN LINEAL ===")

    # 1. Preparación de datos
    df = df.copy()
    
    # Verificar columnas disponibles
    print("Columnas disponibles:", df.columns.tolist())

    # Seleccionar características (usaremos 'year' como variable temporal)
    if 'year' not in df.columns:
        print("⚠️ Error: Se requiere una columna 'year' para el análisis de series de tiempo.")
        return df

    # Agrupar datos por año (ej: promedio de 'score' por año)
    time_series = df.groupby('year')['score'].mean().reset_index()
    time_series = time_series.dropna()

    if len(time_series) < 2:
        print("⚠️ Error: No hay suficientes datos temporales para el modelo.")
        return df

    # 2. Crear variables para el modelo
    X = time_series[['year']]  # Variable independiente (año)
    y = time_series['score']   # Variable dependiente (ej: puntuación promedio)

    # Dividir datos en entrenamiento y prueba (80% - 20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Entrenar modelo de regresión lineal
    model = LinearRegression()
    model.fit(X_train, y_train)

    # 4. Evaluar el modelo
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"\n📊 Métricas del modelo:")
    print(f"- Error Cuadrático Medio (MSE): {mse:.2f}")
    print(f"- Coeficiente de Determinación (R²): {r2:.2f}")

    # 5. Predecir años futuros (ej: 3 años adelante)
    future_years = np.arange(X['year'].max() + 1, X['year'].max() + 4).reshape(-1, 1)
    future_predictions = model.predict(future_years)

    print("\n🔮 Predicciones para años futuros:")
    for year, pred in zip(future_years.flatten(), future_predictions):
        print(f"- Año {year}: Puntuación estimada = {pred:.2f}")

    # 6. Visualización
    plt.figure(figsize=(12, 6))
    plt.scatter(X, y, color='blue', label='Datos reales')
    plt.plot(X, model.predict(X), color='red', label='Regresión lineal')
    plt.scatter(future_years, future_predictions, color='green', marker='X', s=100, label='Predicciones futuras')
    plt.xlabel('Año')
    plt.ylabel('Puntuación promedio')
    plt.title('Forecasting de Puntuaciones de Videojuegos')
    plt.legend()
    plt.grid(True)
    plt.savefig('forecasting.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("\n🎉 ¡PRÁCTICA 8 COMPLETADA!")
    return df