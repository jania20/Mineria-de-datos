from practica_1 import practica_1
from practica_2 import practica_2
from practica_3 import practica_3
from practica_4 import practica_4
from practica_5 import practica_5
from practica_6 import practica_6
from practica_7 import practica_7
from practica_8 import practica_8
from practica_9 import practica_9 

def main():
    print("=== EJECUCIÓN DE TODAS LAS PRÁCTICAS ===")
    df = practica_1()  # Carga y limpia datos
    df = practica_2(df)  # Análisis exploratorio
    df = practica_3(df)  # Visualizaciones
    df = practica_4(df)  # Análisis estadístico
    df = practica_5(df)  # Modelos lineales y correlación
    df = practica_6(df)  # Clasificación con KNN
    df = practica_7(df)  # Clustering con K-Means
    df = practica_8(df)  # Forecasting con regresión lineal
    df =  practica_9(df)  # Word Cloud
    print("\n🎉 TODAS LAS PRÁCTICAS COMPLETADAS CON ÉXITO")

if __name__ == "__main__":
    main()