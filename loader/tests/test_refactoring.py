"""
Test básico de la refactorización del loader
"""


def test_basic_imports():
    """Prueba que todos los imports principales funcionen"""
    print("🔍 Testing imports básicos...")

    try:
        # Test core imports
        from loader.core import ENAHOConfig, ENAHOError, setup_logging, CacheManager
        print("✅ Core imports: OK")

        # Test IO imports
        from loader.io import ENAHODataDownloader, ENAHOLocalReader, ReaderFactory
        print("✅ IO imports: OK")

        # Test utils imports
        from loader.utils import download_enaho_data, read_enaho_file, ENAHOUtils
        print("✅ Utils imports: OK")

        # Test específicos de readers
        from loader.io.readers import SPSSReader, StataReader, ParquetReader, CSVReader
        print("✅ Readers imports: OK")

        print("🎉 Todos los imports funcionan correctamente!")
        return True

    except ImportError as e:
        print(f"❌ Error de import: {e}")
        return False
    except Exception as e:
        print(f"❌ Error inesperado: {e}")
        return False


def test_config_creation():
    """Prueba la creación de configuración"""
    print("\n🔍 Testing configuración...")

    try:
        from loader.core import ENAHOConfig

        # Crear config por defecto
        config = ENAHOConfig()
        print(f"✅ Config creada: {config.base_url}")
        print(f"✅ Módulos disponibles: {len(config.AVAILABLE_MODULES)}")
        print(f"✅ Años transversales: {len(config.YEAR_MAP_TRANSVERSAL)}")

        return True
    except Exception as e:
        print(f"❌ Error en configuración: {e}")
        return False


def test_downloader_creation():
    """Prueba la creación del downloader principal"""
    print("\n🔍 Testing creación de downloader...")

    try:
        from loader.io import ENAHODataDownloader

        downloader = ENAHODataDownloader(verbose=False)

        # Probar métodos básicos
        years = downloader.get_available_years()
        modules = downloader.get_available_modules()

        print(f"✅ Downloader creado")
        print(f"✅ Años disponibles: {len(years)}")
        print(f"✅ Módulos disponibles: {len(modules)}")

        return True
    except Exception as e:
        print(f"❌ Error creando downloader: {e}")
        return False


def test_utility_functions():
    """Prueba las funciones de utilidad"""
    print("\n🔍 Testing funciones de utilidad...")

    try:
        from loader.utils import get_available_data, validate_download_request, ENAHOUtils

        # Test disponibilidad de datos
        data_info = get_available_data(is_panel=False)
        print(f"✅ Info de datos: {data_info['dataset_type']}")

        # Test validación
        validation = validate_download_request(["01", "34"], ["2023", "2022"])
        print(f"✅ Validación: {validation['status']}")

        # Test utilidades
        estimate = ENAHOUtils.estimate_download_size(["01", "34"], ["2023"])
        print(f"✅ Estimación descarga: {estimate['total_mb']:.1f} MB")

        return True
    except Exception as e:
        print(f"❌ Error en utilidades: {e}")
        return False


if __name__ == "__main__":
    print("🚀 Iniciando tests de refactorización...\n")

    tests = [
        test_basic_imports,
        test_config_creation,
        test_downloader_creation,
        test_utility_functions
    ]

    results = []
    for test in tests:
        results.append(test())

    print(f"\n📊 Resultados: {sum(results)}/{len(results)} tests pasaron")

    if all(results):
        print("🎉 ¡Refactorización exitosa! Todo funciona correctamente.")
    else:
        print("⚠️  Hay algunos problemas que revisar.")