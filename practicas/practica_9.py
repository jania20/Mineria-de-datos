# ------------------------- PRACTICA 9: WORD CLOUD -----------------------------------
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter

def practica_9(df=None, text_column="name"):
    """
    Genera una nube de palabras (word cloud) a partir de una columna de texto del dataset.
    
    Parámetros:
        df (DataFrame): DataFrame de pandas (opcional, si no se proporciona, se carga el CSV limpio).
        text_column (str): Columna con el texto a analizar (ej. 'name', 'genre', 'publisher').
    """
    # Cargar datos si no se proporcionan
    if df is None:
        try:
            df = pd.read_csv('./cleaned_games_data.csv')
        except FileNotFoundError:
            print("Error: No se encontró 'cleaned_games_data.csv'")
            return

    # Verificar que la columna exista
    if text_column not in df.columns:
        print(f"Error: La columna '{text_column}' no existe en el dataset.")
        return

    # Obtener texto y limpiar
    text_data = " ".join(str(item) for item in df[text_column].dropna())

    # Contar frecuencias de palabras 
    word_freq = Counter(text_data.split()).most_common(20)
    print("\nTop 20 palabras más frecuentes:")
    for word, count in word_freq:
        print(f"{word}: {count}")

    
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white',
        colormap='viridis', 
        max_words=100,
        stopwords=None  
    ).generate(text_data)

    # Mostrar y guardar
    plt.figure(figsize=(12, 8))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f"Word Cloud - Columna: '{text_column}'", pad=20)
    plt.tight_layout()
    
    # Guardar imagen
    output_file = f"wordcloud_{text_column}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n✅ Word cloud guardado como '{output_file}'")


if __name__ == "__main__":
    practica_9(text_column="name")  # Cambia a 'genre' o 'platform' si prefieres