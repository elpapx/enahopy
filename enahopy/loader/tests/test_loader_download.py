#!/usr/bin/env python3
"""
Script de prueba de descarga real para el módulo loader
ADVERTENCIA: Este script descarga datos reales del INEI
"""

import sys
import time
import shutil
from pathlib import Path
from datetime import datetime

# Agregar el directorio del proyecto al path
project_dir = Path(__file__).parent.parent  # Ajustar según ubicación
sys.path.insert(0, str(project_dir))


def confirm_download():
    """Solicitar confirmación antes de descargar"""
    print("\n" + "⚠️  ADVERTENCIA ".center(60, "="))
    print("Este script descargará datos reales del INEI")
    print("Esto puede tomar tiempo y usar ancho de banda")
    print("=" * 60)

    response = input("\n¿Desea continuar? (s/n): ").strip().lower()
    return response == 's'


def test_single_module_download():
    """Descargar un solo módulo pequeño para prueba"""
    from loader.utils import download_enaho_data, ENAHOUtils

    print("\n" + "=" * 60)
    print("TEST: DESCARGA DE MÓDULO INDIVIDUAL")
    print("=" * 60)

    # Usar módulo 37 (Gobierno Electrónico) que es pequeño
    module = "37"
    year = "2023"
    output_dir = "./test_downloads"

    # Estimar tamaño
    estimate = ENAHOUtils.estimate_download_size([module], [year])
    print(f"\n📊 Estimación de descarga:")
    print(f"   - Módulo: {module} ({ENAHOUtils.get_module_description(module)})")
    print(f"   - Año: {year}")
    print(f"   - Tamaño estimado: {estimate['total_mb']:.1f} MB")
    print(f"   - Directorio: {output_dir}")

    try:
        print("\n⏳ Iniciando descarga...")
        start_time = time.time()

        # Callback para mostrar progreso
        def progress_callback(task_name, completed, total):
            percent = (completed / total) * 100
            print(f"   📥 Progreso: {completed}/{total} ({percent:.0f}%) - {task_name}")

        # Descargar solo el ZIP
        result = download_enaho_data(
            modules=[module],
            years=[year],
            output_dir=output_dir,
            decompress=False,  # No descomprimir
            only_dta=False,
            load_dta=False,
            overwrite=True,
            parallel=False,
            verbose=True,
            progress_callback=progress_callback
        )

        elapsed = time.time() - start_time
        print(f"\n✅ Descarga completada en {elapsed:.1f} segundos")

        # Verificar archivo descargado
        output_path = Path(output_dir)
        if output_path.exists():
            zip_files = list(output_path.glob("*.zip"))
            print(f"\n📦 Archivos descargados:")
            for zip_file in zip_files:
                size_mb = zip_file.stat().st_size / (1024 * 1024)
                print(f"   - {zip_file.name}: {size_mb:.1f} MB")

        return True

    except Exception as e:
        print(f"\n❌ Error en descarga: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_download_and_extract():
    """Descargar y extraer archivos .dta"""
    from loader.utils import download_enaho_data

    print("\n" + "=" * 60)
    print("TEST: DESCARGA Y EXTRACCIÓN")
    print("=" * 60)

    module = "37"
    year = "2023"
    output_dir = "./test_downloads"

    try:
        print("\n⏳ Descargando y extrayendo...")

        result = download_enaho_data(
            modules=[module],
            years=[year],
            output_dir=output_dir,
            decompress=True,  # Descomprimir
            only_dta=True,  # Solo archivos .dta
            load_dta=False,  # No cargar en memoria
            overwrite=True,
            verbose=True
        )

        print("\n✅ Extracción completada")

        # Verificar archivos extraídos
        extract_dir = Path(output_dir) / f"modulo_{module}_{year}"
        if extract_dir.exists():
            dta_files = list(extract_dir.glob("*.dta"))
            print(f"\n📊 Archivos .dta extraídos: {len(dta_files)}")
            for dta_file in dta_files[:5]:  # Mostrar máximo 5
                size_kb = dta_file.stat().st_size / 1024
                print(f"   - {dta_file.name}: {size_kb:.1f} KB")

        return True

    except Exception as e:
        print(f"\n❌ Error: {e}")
        return False


def test_parallel_download():
    """Probar descarga paralela de múltiples módulos"""
    from loader.utils import download_enaho_data, ENAHOUtils

    print("\n" + "=" * 60)
    print("TEST: DESCARGA PARALELA")
    print("=" * 60)

    # Usar módulos pequeños
    modules = ["37", "09"]  # Gobierno Electrónico y Gastos del Hogar
    year = "2023"
    output_dir = "./test_downloads_parallel"

    # Estimación
    estimate = ENAHOUtils.estimate_download_size(modules, [year])
    print(f"\n📊 Estimación total: {estimate['total_mb']:.1f} MB")

    # Recomendación de paralelización
    rec = ENAHOUtils.recommend_parallel_settings(len(modules))
    print(f"⚡ Configuración recomendada: {rec['max_workers']} workers")

    try:
        print("\n⏳ Iniciando descarga paralela...")
        start_time = time.time()

        downloads_completed = []

        def progress_callback(task_name, completed, total):
            downloads_completed.append(task_name)
            print(f"   ✅ Completado: {task_name} ({completed}/{total})")

        result = download_enaho_data(
            modules=modules,
            years=[year],
            output_dir=output_dir,
            decompress=False,
            parallel=True,
            max_workers=rec['max_workers'],
            verbose=True,
            progress_callback=progress_callback
        )

        elapsed = time.time() - start_time
        print(f"\n✅ Descarga paralela completada en {elapsed:.1f} segundos")
        print(f"📦 Módulos descargados: {downloads_completed}")

        return True

    except Exception as e:
        print(f"\n❌ Error: {e}")
        return False


def test_load_to_dataframe():
    """Cargar datos directamente a DataFrames"""
    from loader.utils import download_enaho_data
    import pandas as pd

    print("\n" + "=" * 60)
    print("TEST: CARGA A DATAFRAME")
    print("=" * 60)

    module = "37"
    year = "2023"
    output_dir = "./test_downloads"

    try:
        print("\n⏳ Descargando y cargando a DataFrame...")

        result = download_enaho_data(
            modules=[module],
            years=[year],
            output_dir=output_dir,
            decompress=True,
            only_dta=True,
            load_dta=True,  # Cargar en memoria
            overwrite=True,
            verbose=True,
            low_memory=True  # Optimizar memoria
        )

        if result:
            print("\n✅ Datos cargados en DataFrames")

            for (year_key, module_key), dataframes in result.items():
                print(f"\n📊 Año {year_key}, Módulo {module_key}:")

                for filename, df in list(dataframes.items())[:3]:  # Máximo 3
                    print(f"\n   📄 {filename}:")
                    print(f"      - Filas: {len(df):,}")
                    print(f"      - Columnas: {len(df.columns)}")
                    print(f"      - Memoria: {df.memory_usage(deep=True).sum() / 1024 ** 2:.1f} MB")
                    print(f"      - Columnas: {list(df.columns[:5])}...")

                    # Mostrar primeras filas
                    if len(df) > 0:
                        print(f"\n      Primeras 3 filas:")
                        print(df.head(3).to_string(max_cols=5))

        return True

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def cleanup_test_files():
    """Limpiar archivos de prueba"""
    print("\n" + "=" * 60)
    print("LIMPIEZA DE ARCHIVOS DE PRUEBA")
    print("=" * 60)

    dirs_to_clean = ["./test_downloads", "./test_downloads_parallel"]

    for dir_path in dirs_to_clean:
        path = Path(dir_path)
        if path.exists():
            try:
                shutil.rmtree(path)
                print(f"✅ Eliminado: {dir_path}")
            except Exception as e:
                print(f"⚠️  No se pudo eliminar {dir_path}: {e}")
        else:
            print(f"ℹ️  No existe: {dir_path}")


def main():
    """Ejecutar pruebas de descarga"""
    print("\n" + "🚀 PRUEBAS DE DESCARGA REAL ".center(60, "="))

    if not confirm_download():
        print("❌ Prueba cancelada por el usuario")
        return False

    # Crear directorio de pruebas
    Path("./test_downloads").mkdir(exist_ok=True)

    tests = [
        ("Descarga Individual", test_single_module_download),
        ("Descarga y Extracción", test_download_and_extract),
        ("Descarga Paralela", test_parallel_download),
        ("Carga a DataFrame", test_load_to_dataframe)
    ]

    results = []
    for name, test_func in tests:
        print(f"\n🔄 Ejecutando: {name}")
        try:
            success = test_func()
            results.append((name, success))
            if not success:
                print(f"⚠️  Test '{name}' falló, continuando...")
        except KeyboardInterrupt:
            print("\n⚠️  Prueba interrumpida por el usuario")
            break
        except Exception as e:
            print(f"\n❌ Error no manejado en {name}: {e}")
            results.append((name, False))

    # Resumen
    print("\n" + "=" * 60)
    print("RESUMEN DE PRUEBAS DE DESCARGA")
    print("=" * 60)

    for name, success in results:
        status = "✅" if success else "❌"
        print(f"{status} {name}")

    # Preguntar si limpiar archivos
    print("\n" + "=" * 60)
    response = input("\n¿Desea eliminar los archivos de prueba? (s/n): ").strip().lower()
    if response == 's':
        cleanup_test_files()
    else:
        print("ℹ️  Archivos de prueba conservados")

    passed = sum(1 for _, s in results if s)
    print(f"\n📊 Resultado final: {passed}/{len(results)} pruebas pasaron")

    return passed == len(results)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)