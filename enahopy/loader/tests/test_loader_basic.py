#!/usr/bin/env python3
"""
Script de prueba básica para el módulo loader de enahopy
Prueba funcionalidades esenciales sin realizar descargas reales
"""

import sys
from pathlib import Path

# Agregar el directorio del proyecto al path
project_dir = Path(__file__).parent.parent  # Ajustar según ubicación
sys.path.insert(0, str(project_dir))


def test_imports():
    """Test 1: Verificar que todos los imports funcionan"""
    print("=" * 60)
    print("TEST 1: VERIFICACIÓN DE IMPORTS")
    print("=" * 60)

    try:
        # Core imports
        from loader.core import ENAHOConfig, ENAHOError, setup_logging, CacheManager
        print("✅ Core imports OK")

        # IO imports
        from loader.io import ENAHODataDownloader, ENAHOLocalReader
        print("✅ IO imports OK")

        # Utils imports
        from loader.utils import download_enaho_data, read_enaho_file, ENAHOUtils
        print("✅ Utils imports OK")

        # Validators imports
        from loader.io.validators import ENAHOValidator
        print("✅ Validators imports OK")

        return True

    except ImportError as e:
        print(f"❌ Error de importación: {e}")
        return False
    except Exception as e:
        print(f"❌ Error inesperado: {e}")
        return False


def test_configuration():
    """Test 2: Verificar la configuración"""
    print("\n" + "=" * 60)
    print("TEST 2: CONFIGURACIÓN")
    print("=" * 60)

    try:
        from loader.core import ENAHOConfig

        # Crear configuración por defecto
        config = ENAHOConfig()

        print(f"📍 URL Base: {config.base_url}")
        print(f"📂 Directorio de cache: {config.cache_dir}")
        print(f"⏱️  TTL del cache: {config.cache_ttl_hours} horas")
        print(f"👥 Max workers por defecto: {config.default_max_workers}")

        # Verificar módulos disponibles
        print(f"\n📦 Módulos disponibles: {len(config.AVAILABLE_MODULES)}")
        for code, desc in list(config.AVAILABLE_MODULES.items())[:3]:
            print(f"   - {code}: {desc[:50]}...")

        # Verificar años disponibles
        print(f"\n📅 Años transversales: {len(config.YEAR_MAP_TRANSVERSAL)}")
        years_trans = sorted(config.YEAR_MAP_TRANSVERSAL.keys(), reverse=True)[:5]
        print(f"   Últimos años: {years_trans}")

        print(f"\n📅 Años panel: {len(config.YEAR_MAP_PANEL)}")
        years_panel = sorted(config.YEAR_MAP_PANEL.keys(), reverse=True)[:5]
        print(f"   Últimos años: {years_panel}")

        return True

    except Exception as e:
        print(f"❌ Error en configuración: {e}")
        return False


def test_downloader_creation():
    """Test 3: Crear instancia del downloader"""
    print("\n" + "=" * 60)
    print("TEST 3: CREACIÓN DEL DOWNLOADER")
    print("=" * 60)

    try:
        from loader.io import ENAHODataDownloader

        # Crear downloader sin verbose para evitar logs
        downloader = ENAHODataDownloader(verbose=False)
        print("✅ Downloader creado exitosamente")

        # Obtener información disponible
        years = downloader.get_available_years(is_panel=False)
        modules = downloader.get_available_modules()

        print(f"📅 Años disponibles: {len(years)}")
        print(f"   Rango: {years[-1]} - {years[0]}")

        print(f"📦 Módulos disponibles: {len(modules)}")
        module_list = list(modules.keys())[:5]
        print(f"   Primeros módulos: {module_list}")

        return True

    except Exception as e:
        print(f"❌ Error creando downloader: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_validation():
    """Test 4: Validación de parámetros"""
    print("\n" + "=" * 60)
    print("TEST 4: VALIDACIÓN DE PARÁMETROS")
    print("=" * 60)

    try:
        from loader.io import ENAHODataDownloader

        downloader = ENAHODataDownloader(verbose=False)

        # Test 1: Validación exitosa
        print("\n🔍 Validación con parámetros correctos:")
        result = downloader.validate_availability(
            modules=["01", "34"],
            years=["2023", "2022"],
            is_panel=False
        )

        if result['status'] == 'valid':
            print("✅ Validación exitosa")
            print(f"   - Módulos: {result['modules']}")
            print(f"   - Años: {result['years']}")
            print(f"   - Tipo: {result['dataset_type']}")
            print(f"   - Descargas estimadas: {result['estimated_downloads']}")
        else:
            print(f"❌ Validación falló: {result.get('error', 'Error desconocido')}")

        # Test 2: Validación con error (módulo inválido)
        print("\n🔍 Validación con módulo inválido:")
        result = downloader.validate_availability(
            modules=["99"],  # Módulo que no existe
            years=["2023"],
            is_panel=False
        )

        if result['status'] == 'invalid':
            print("✅ Error detectado correctamente")
            print(f"   - Error: {result.get('error', 'N/A')}")
        else:
            print("❌ Debería haber detectado el error")

        # Test 3: Validación con año inválido
        print("\n🔍 Validación con año inválido:")
        result = downloader.validate_availability(
            modules=["01"],
            years=["1990"],  # Año muy antiguo
            is_panel=False
        )

        if result['status'] == 'invalid':
            print("✅ Error detectado correctamente")
            print(f"   - Error: {result.get('error', 'N/A')}")
        else:
            print("❌ Debería haber detectado el error")

        return True

    except Exception as e:
        print(f"❌ Error en validación: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_utilities():
    """Test 5: Funciones de utilidad"""
    print("\n" + "=" * 60)
    print("TEST 5: FUNCIONES DE UTILIDAD")
    print("=" * 60)

    try:
        from loader.utils import ENAHOUtils, get_available_data, validate_download_request

        # Test 1: Obtener datos disponibles
        print("\n📊 Información de datos disponibles:")
        data_info = get_available_data(is_panel=False)
        print(f"✅ Tipo de dataset: {data_info['dataset_type']}")
        print(f"   - Años disponibles: {len(data_info['years'])}")
        print(f"   - Módulos disponibles: {len(data_info['modules'])}")

        # Test 2: Estimación de tamaño
        print("\n💾 Estimación de tamaño de descarga:")
        estimate = ENAHOUtils.estimate_download_size(
            modules=["01", "34"],
            years=["2023", "2022"]
        )
        print(f"✅ Tamaño total estimado: {estimate['total_mb']:.1f} MB")
        print(f"   - En GB: {estimate['total_gb']:.2f} GB")
        print(f"   - Comprimido: ~{estimate['compressed_size']:.1f} MB")

        # Test 3: Descripción de módulos
        print("\n📝 Descripción de módulos:")
        for module in ["01", "34", "37"]:
            desc = ENAHOUtils.get_module_description(module)
            print(f"✅ Módulo {module}: {desc[:60]}...")

        # Test 4: Recomendación de paralelización
        print("\n⚡ Recomendaciones de paralelización:")
        for num in [2, 5, 10, 25]:
            rec = ENAHOUtils.recommend_parallel_settings(num)
            print(f"✅ Para {num} descargas: {rec['max_workers']} workers ({rec['reason']})")

        return True

    except Exception as e:
        print(f"❌ Error en utilidades: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Ejecutar todas las pruebas"""
    print("\n" + "🚀 INICIANDO PRUEBAS DEL MÓDULO LOADER ".center(60, "="))
    print("=" * 60)

    tests = [
        ("Imports", test_imports),
        ("Configuración", test_configuration),
        ("Creación Downloader", test_downloader_creation),
        ("Validación", test_validation),
        ("Utilidades", test_utilities)
    ]

    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"\n❌ Error ejecutando test {name}: {e}")
            results.append((name, False))

    # Resumen final
    print("\n" + "=" * 60)
    print("RESUMEN DE PRUEBAS")
    print("=" * 60)

    passed = sum(1 for _, success in results if success)
    total = len(results)

    for name, success in results:
        status = "✅ PASÓ" if success else "❌ FALLÓ"
        print(f"{status} - {name}")

    print(f"\n📊 Resultado: {passed}/{total} pruebas pasaron")

    if passed == total:
        print("🎉 ¡Todas las pruebas pasaron exitosamente!")
    else:
        print(f"⚠️  {total - passed} pruebas fallaron")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)