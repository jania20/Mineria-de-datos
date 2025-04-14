import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from yellowbrick.cluster import KElbowVisualizer

def practica_7(df):
    """
    Implementa un modelo de clustering con K-Means para agrupar videojuegos
    basado en sus características numéricas.
    
    Args:
        df (DataFrame): DataFrame con los datos de videojuegos
        
    Returns:
        DataFrame: DataFrame original con una columna adicional para los clusters
    """
    print("\n=== PRÁCTICA 7: CLUSTERING CON K-MEANS ===")
    
    # 1. Preparación de datos
    df = df.copy()
    
    # Verificar columnas disponibles
    print("Columnas disponibles:", df.columns.tolist())
    
    # Seleccionar características numéricas para clustering
    numeric_features = ['score', 'year']  # Ajustar según tus columnas disponibles
    use_features = [col for col in numeric_features if col in df.columns]
    
    if len(use_features) < 2:
        print("⚠️ No hay suficientes características numéricas para clustering")
        return df
    
    # Filtrar y limpiar datos
    X = df[use_features].dropna()
    
    if len(X) == 0:
        print("⚠️ No hay datos válidos después de la limpieza")
        return df
    
    # Escalar los datos (K-Means es sensible a la escala)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 2. Determinar el número óptimo de clusters
    print("\n🔍 DETERMINANDO NÚMERO ÓPTIMO DE CLUSTERS")
    
    # Método del codo
    plt.figure(figsize=(10, 6))
    visualizer = KElbowVisualizer(KMeans(random_state=42), k=(2, 10))
    visualizer.fit(X_scaled)
    visualizer.show(outpath="elbow_method.png")
    plt.close()
    optimal_k = visualizer.elbow_value_
    print(f"Número óptimo de clusters (método del codo): {optimal_k}")
    
    # Método de la silueta
    if optimal_k is not None:
        range_n_clusters = range(2, min(10, len(X)+1))
        silhouette_scores = []
        
        for n_clusters in range_n_clusters:
            clusterer = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = clusterer.fit_predict(X_scaled)
            silhouette_avg = silhouette_score(X_scaled, cluster_labels)
            silhouette_scores.append(silhouette_avg)
        
        plt.figure(figsize=(10, 6))
        plt.plot(range_n_clusters, silhouette_scores, 'bo-')
        plt.xlabel('Número de clusters')
        plt.ylabel('Puntaje de Silueta')
        plt.title('Método de la Silueta para K óptimo')
        plt.savefig('silhouette_method.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        optimal_k_silhouette = range_n_clusters[np.argmax(silhouette_scores)]
        print(f"Número óptimo de clusters (método silueta): {optimal_k_silhouette}")
        
        # Usar el óptimo de ambos métodos
        final_k = optimal_k if optimal_k == optimal_k_silhouette else optimal_k_silhouette
    else:
        final_k = 3  # Valor por defecto si no se puede determinar
    
    print(f"\n🔢 Usando {final_k} clusters para el modelo final")
    
    # 3. Crear y entrenar modelo K-Means
    print("\n🔍 ENTRENANDO MODELO K-MEANS...")
    kmeans = KMeans(n_clusters=final_k, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)
    
    # 4. Analizar resultados
    df_clustered = df.copy()
    df_clustered['cluster'] = np.nan
    df_clustered.loc[X.index, 'cluster'] = clusters
    
    # Estadísticas por cluster
    cluster_stats = df_clustered.groupby('cluster')[use_features].mean()
    print("\n📊 ESTADÍSTICAS POR CLUSTER:")
    print(cluster_stats)
    
    # 5. Visualización de clusters
    print("\n📊 VISUALIZANDO RESULTADOS...")
    plt.figure(figsize=(12, 8))
    
    if len(use_features) == 2:
        # Gráfico 2D si tenemos exactamente 2 características
        sns.scatterplot(
            x=use_features[0], 
            y=use_features[1], 
            hue='cluster', 
            palette='viridis', 
            data=df_clustered,
            legend='full'
        )
        plt.scatter(
            kmeans.cluster_centers_[:, 0], 
            kmeans.cluster_centers_[:, 1],
            s=300, c='red', marker='X'
        )
    else:
        # Gráfico de las dos primeras características si hay más de 2
        sns.scatterplot(
            x=use_features[0], 
            y=use_features[1], 
            hue='cluster', 
            palette='viridis', 
            data=df_clustered,
            legend='full'
        )
    
    plt.title(f'Clusters de Videojuegos (K={final_k})', fontsize=14)
    plt.savefig('kmeans_clusters.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("📌 Gráfico guardado: 'kmeans_clusters.png'")
    
    # 6. Interpretación de clusters
    print("\n🔍 INTERPRETACIÓN DE CLUSTERS:")
    for i in range(final_k):
        cluster_data = df_clustered[df_clustered['cluster'] == i]
        print(f"\n📌 Cluster {i} ({len(cluster_data)} juegos):")
        print(f"- Géneros más comunes: {cluster_data['genre'].value_counts().head(3).to_dict()}")
        print(f"- Plataformas más comunes: {cluster_data['platform'].value_counts().head(3).to_dict()}")
        print(f"- Score promedio: {cluster_data['score'].mean():.2f}")
        if 'year' in cluster_data.columns:
            print(f"- Año promedio: {cluster_data['year'].mean():.2f}")
    
    print("\n🎉 ¡PRÁCTICA 7 COMPLETADA!")
    return df_clustered