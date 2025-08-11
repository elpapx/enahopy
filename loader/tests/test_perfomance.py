"""
Test de performance de la refactorización
"""
import time
from pathlib import Path


def test_import_performance():
    """Mide tiempo de imports"""
    print("⏱️  Testing performance de imports...")

    start_time = time.time()

    from loader.io import ENAHODataDownloader
    from loader.utils import read_enaho_file, ENAHOUtils

    import_time = time.time() - start_time

    print(f"✅ Imports completados en: {import_time:.3f}s")

    return import_time < 5.0  # Debe ser rápido


def test_reading_performance():
    """Mide performance de lectura"""
    print("\n⏱️  Testing performance de lectura...")

    dta_files = list(Path("./dta").glob("*.dta"))

    if not dta_files:
        print("ℹ️  No hay archivos para test de performance")
        return True

    test_file = dta_files[0]
    print(f"📊 Archivo test: {test_file.name}")

    try:
        from loader.utils import read_enaho_file

        # Lectura normal
        start_time = time.time()
        data, validation = read_enaho_file(
            str(test_file),
            columns=['conglome', 'vivienda', 'hogar'],
            verbose=False
        )
        read_time = time.time() - start_time

        print(f"✅ Lectura de {len(data)} filas en: {read_time:.3f}s")
        print(f"✅ Performance: {len(data) / read_time:.0f} filas/segundo")

        return True

    except Exception as e:
        print(f"❌ Error en performance: {e}")
        return False


if __name__ == "__main__":
    print("🚀 TESTING PERFORMANCE")
    print("=" * 30)

    tests = [test_import_performance, test_reading_performance]
    results = [test() for test in tests]

    print(f"\n📊 Performance: {sum(results)}/{len(results)} tests OK")