"""
Test con archivo real si existe
"""
import os
from pathlib import Path


def test_with_real_file():
    """Prueba con archivo real si existe"""
    print("🔍 Buscando archivos de prueba...")

    # Buscar archivos de prueba en directorios comunes
    test_dirs = ["./data", "./dta", "./test_data", "./ejemplos"]
    test_files = []

    for test_dir in test_dirs:
        if Path(test_dir).exists():
            # Buscar archivos .dta, .sav, .csv
            for pattern in ["*.dta", "*.sav", "*.csv"]:
                test_files.extend(Path(test_dir).glob(pattern))

    if not test_files:
        print("ℹ️  No se encontraron archivos de prueba")
        print("💡 Puedes crear un archivo CSV simple para probar:")
        print("""
# Crear archivo de prueba simple:
import pandas as pd
test_df = pd.DataFrame({
    'conglome': [1, 2, 3],
    'vivienda': [1, 1, 2], 
    'hogar': [1, 1, 1],
    'variable1': [10, 20, 30]
})
test_df.to_csv('./test_data.csv', index=False)
        """)
        return False

    print(f"📁 Encontrados {len(test_files)} archivos de prueba")

    try:
        from loader.utils import read_enaho_file, get_file_info

        # Probar con el primer archivo encontrado
        test_file = test_files[0]
        print(f"🧪 Probando con: {test_file}")

        # Test obtener información
        file_info = get_file_info(str(test_file), verbose=False)
        print(f"✅ Info archivo: {file_info['total_columns']} columnas")

        # Test lectura (solo primeras 3 columnas para ser seguro)
        available_cols = list(file_info['sample_columns'])[:3]

        if available_cols:
            data, validation = read_enaho_file(
                str(test_file),
                columns=available_cols,
                verbose=False
            )
            print(f"✅ Lectura exitosa: {len(data)} filas, {len(data.columns)} columnas")
            print(f"✅ Validación: {validation.get_summary()}")

        return True

    except Exception as e:
        print(f"❌ Error con archivo real: {e}")
        return False


if __name__ == "__main__":
    test_with_real_file()