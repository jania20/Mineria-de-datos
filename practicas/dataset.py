from practica_1 import practica_1
from practica_2 import practica_2
from practica_3 import practica_3
from practica_4 import practica_4

def main():
    print("=== EJECUCIÓN DE TODAS LAS PRÁCTICAS ===")
    df = practica_1()  # Carga y limpia datos
    df = practica_2(df)  # Análisis exploratorio
    df = practica_3(df)  # Visualizaciones
    df = practica_4(df)  # Análisis estadístico
    print("\n🎉 TODAS LAS PRÁCTICAS COMPLETADAS CON ÉXITO")

if __name__ == "__main__":
    main()