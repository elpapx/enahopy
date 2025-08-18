"""
Test de funcionalidades avanzadas de la refactorización
"""


def test_multiple_file_reading():
    """Prueba lectura de múltiples archivos"""
    print("🔍 Testing lectura múltiple de archivos...")

    try:
        from loader.io import ENAHODataDownloader
        from pathlib import Path

        downloader = ENAHODataDownloader(verbose=False)

        # Buscar archivos DTA
        dta_files = list(Path("./dta").glob("*.dta"))[:3]  # Solo primeros 3

        if len(dta_files) >= 2:
            print(f"📁 Probando con {len(dta_files)} archivos")

            # Test batch reading
            results = downloader.batch_read_local_files(
                dta_files,
                columns=['conglome', 'vivienda', 'hogar'],
                ignore_missing_columns=True
            )

            print(f"✅ Lectura en lote: {len(results)} archivos procesados")

            for filename, (data, validation) in results.items():
                print(f"  - {filename}: {len(data)} filas")

            return True
        else:
            print("ℹ️  Necesitas al menos 2 archivos .dta para esta prueba")
            return True  # No es un error

    except Exception as e:
        print(f"❌ Error en lectura múltiple: {e}")
        return False


def test_metadata_extraction():
    """Prueba extracción de metadatos"""
    print("\n🔍 Testing extracción de metadatos...")

    try:
        from loader.utils import get_file_info
        from pathlib import Path

        # Usar el primer archivo .dta encontrado
        dta_files = list(Path("./dta").glob("*.dta"))

        if dta_files:
            test_file = dta_files[0]
            print(f"📊 Extrayendo metadatos de: {test_file.name}")

            # Información básica
            file_info = get_file_info(str(test_file), verbose=False)

            print(f"✅ Archivo: {file_info['file_info']['file_format']}")
            print(f"✅ Columnas totales: {file_info['total_columns']}")
            print(f"✅ Tiene etiquetas: {file_info['has_labels']}")

            # Metadatos completos
            from loader.io import ENAHOLocalReader
            reader = ENAHOLocalReader(str(test_file), verbose=False)

            full_metadata = reader.extract_metadata()
            print(f"✅ Metadatos completos extraídos: {len(full_metadata)} secciones")

            return True
        else:
            print("ℹ️  No se encontraron archivos .dta")
            return True

    except Exception as e:
        print(f"❌ Error extrayendo metadatos: {e}")
        return False


def test_chunked_reading():
    """Prueba lectura por chunks"""
    print("\n🔍 Testing lectura por chunks...")

    try:
        from loader.utils import read_enaho_file
        from pathlib import Path

        dta_files = list(Path("./dta").glob("*.dta"))

        if dta_files:
            test_file = dta_files[0]
            print(f"📦 Lectura por chunks: {test_file.name}")

            # Leer por chunks
            data_chunks, validation = read_enaho_file(
                str(test_file),
                columns=['conglome', 'vivienda', 'hogar'],
                use_chunks=True,
                chunk_size=5000,
                verbose=False
            )

            print(f"✅ Lectura por chunks configurada")
            print(f"✅ Validación: {len(validation.found_columns)} columnas encontradas")

            # Si es iterador, probar primera chunk
            if hasattr(data_chunks, '__iter__') and not hasattr(data_chunks, 'columns'):
                first_chunk = next(iter(data_chunks))
                print(f"✅ Primera chunk: {len(first_chunk)} filas")
            else:
                print(f"✅ Datos cargados: {len(data_chunks)} filas")

            return True
        else:
            print("ℹ️  No se encontraron archivos .dta")
            return True

    except Exception as e:
        print(f"❌ Error en lectura por chunks: {e}")
        return False


def test_validation_and_utilities():
    """Prueba validaciones y utilidades"""
    print("\n🔍 Testing validaciones y utilidades...")

    try:
        from loader.utils import validate_download_request, ENAHOUtils

        # Test validaciones
        print("🔧 Probando validaciones...")

        # Validación exitosa
        valid_request = validate_download_request(["01", "34"], ["2023", "2022"])
        assert valid_request['status'] == 'valid'
        print("✅ Validación exitosa: módulos y años válidos")

        # Validación con error
        invalid_request = validate_download_request(["99"], ["1990"])
        assert invalid_request['status'] == 'invalid'
        print("✅ Validación de error: detecta módulos/años inválidos")

        # Test utilidades
        print("🛠️  Probando utilidades...")

        estimate = ENAHOUtils.estimate_download_size(["01", "34"], ["2023"])
        print(f"✅ Estimación descarga: {estimate['total_mb']:.1f} MB")

        module_desc = ENAHOUtils.get_module_description("01")
        print(f"✅ Descripción módulo: {module_desc[:50]}...")

        parallel_settings = ENAHOUtils.recommend_parallel_settings(10)
        print(f"✅ Recomendación paralelo: {parallel_settings['max_workers']} workers")

        return True

    except Exception as e:
        print(f"❌ Error en validaciones: {e}")
        return False


def test_configuration_and_cache():
    """Prueba configuración y cache"""
    print("\n🔍 Testing configuración y cache...")

    try:
        from loader.core import ENAHOConfig, CacheManager

        # Test configuración
        config = ENAHOConfig()
        print(f"✅ Config URL base: {config.base_url[:50]}...")
        print(f"✅ Módulos disponibles: {len(config.AVAILABLE_MODULES)}")
        print(f"✅ Años transversales: {len(config.YEAR_MAP_TRANSVERSAL)}")
        print(f"✅ Años panel: {len(config.YEAR_MAP_PANEL)}")

        # Test cache
        cache = CacheManager(config.cache_dir)

        # Guardar y recuperar test data
        test_data = {"test": "data", "timestamp": "2023"}
        cache.set_metadata("test_key", test_data)

        retrieved = cache.get_metadata("test_key")
        assert retrieved == test_data
        print("✅ Cache funcionando: guardar y recuperar")

        return True

    except Exception as e:
        print(f"❌ Error en configuración/cache: {e}")
        return False


if __name__ == "__main__":
    print("🚀 TESTING FUNCIONALIDADES AVANZADAS")
    print("=" * 50)

    tests = [
        test_multiple_file_reading,
        test_metadata_extraction,
        test_chunked_reading,
        test_validation_and_utilities,
        test_configuration_and_cache
    ]

    results = []
    for test in tests:
        results.append(test())

    print(f"\n📊 Resultados avanzados: {sum(results)}/{len(results)} tests pasaron")

    if all(results):
        print("🎉 ¡Todas las funcionalidades avanzadas funcionan!")
    else:
        print("⚠️  Revisar funcionalidades con problemas.")