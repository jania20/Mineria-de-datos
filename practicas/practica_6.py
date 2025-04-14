import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

def practica_6(df):
    """
    Versión adaptada que usa solo columnas disponibles
    """
    print("\n=== PRÁCTICA 6: CLASIFICACIÓN CON K-NEAREST NEIGHBORS ===")
    
    # 1. Verificación de columnas necesarias
    required_cols = {'genre', 'score'}
    available_cols = set(df.columns)
    
    if not required_cols.issubset(available_cols):
        missing = required_cols - available_cols
        print(f"⚠️ Columnas faltantes: {missing}. No se puede realizar la práctica 6.")
        return df
    
    #Preparación de datos
    df = df.copy()
    
    #Seleccionar géneros principales
    top_genres = df['genre'].value_counts().nlargest(5).index.tolist()
    df_filtered = df[df['genre'].isin(top_genres)].copy()
    
    #Codificar género
    le = LabelEncoder()
    df_filtered['genre_encoded'] = le.fit_transform(df_filtered['genre'])
    
    #Seleccionar features disponibles
    possible_features = ['score', 'year']  # Añadir más si están en tus datos
    use_features = [col for col in possible_features if col in df_filtered.columns]
    
    if len(use_features) < 1:
        print("⚠️ No hay suficientes features para el modelo")
        return df
    
    X = df_filtered[use_features].fillna(0)
    y = df_filtered['genre_encoded']
    
    #División y escalado
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    #Modelo KNN
    print("\n🔍 ENTRENANDO MODELO KNN...")
    print(f"Usando features: {use_features}")
    print(f"Géneros a predecir: {le.classes_}")
    
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    
    #Evaluación
    y_pred = knn.predict(X_test)
    
    print("\n📊 REPORTE DE CLASIFICACIÓN:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    print(f"\n🔢 EXACTITUD (ACCURACY): {accuracy_score(y_test, y_pred):.2f}")
    
    #Matriz de confusión
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title('Matriz de Confusión - KNN Classifier')
    plt.xlabel('Predicción')
    plt.ylabel('Real')
    plt.tight_layout()
    plt.savefig('knn_confusion_matrix.png', dpi=300)
    plt.close()
    print("\n📌 Gráfico guardado: 'knn_confusion_matrix.png'")
    
    print("\n🎉 ¡PRÁCTICA 6 COMPLETADA!")
    return df