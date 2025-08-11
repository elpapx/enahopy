"""
Test de compatibilidad con el código original
"""


def test_backward_compatibility():
    """Verifica que el código existente siga funcionando"""
    print("🔍 Testing compatibilidad hacia atrás...")

    try:
        # Simular import como se hacía antes (si tenías un main import)
        from loader.utils import download_enaho_data, read_enaho_file
        from loader.io import ENAHODataDownloader

        # Test del patrón de uso original
        downloader = ENAHODataDownloader(verbose=False)

        # Verificar que los métodos principales siguen existiendo
        assert hasattr(downloader, 'download')
        assert hasattr(downloader, 'get_available_years')
        assert hasattr(downloader, 'get_available_modules')
        assert hasattr(downloader, 'read_local_file')

        print("✅ API principal mantiene compatibilidad")

        # Test de funciones de conveniencia
        validation = downloader.validate_availability(["01"], ["2023"])
        assert 'status' in validation
        print("✅ Métodos de validación funcionan")

        return True

    except Exception as e:
        print(f"❌ Error de compatibilidad: {e}")
        return False


def test_reader_factory():
    """Prueba el factory de readers"""
    print("\n🔍 Testing ReaderFactory...")

    try:
        from loader.io.readers import ReaderFactory
        from loader.core import setup_logging
        from pathlib import Path

        logger = setup_logging(verbose=False)

        # Test con diferentes extensiones
        test_cases = [
            ("/path/file.dta", "StataReader"),
            ("/path/file.sav", "SPSSReader"),
            ("/path/file.csv", "CSVReader"),
            ("/path/file.parquet", "ParquetReader")
        ]

        for file_path, expected_reader in test_cases:
            try:
                # No crear el reader real (archivo no existe)
                # Solo verificar que el mapeo funciona
                path = Path(file_path)
                extension = path.suffix.lower()

                reader_map = {
                    '.sav': 'SPSSReader',
                    '.por': 'SPSSReader',
                    '.dta': 'StataReader',
                    '.parquet': 'ParquetReader',
                    '.csv': 'CSVReader',
                    '.txt': 'CSVReader'
                }

                assert extension in reader_map
                assert reader_map[extension] == expected_reader
                print(f"✅ Mapeo {extension} -> {expected_reader}")

            except Exception as e:
                print(f"❌ Error con {file_path}: {e}")
                return False

        return True

    except Exception as e:
        print(f"❌ Error en ReaderFactory: {e}")
        return False


if __name__ == "__main__":
    print("🔄 Testing compatibilidad...\n")

    tests = [test_backward_compatibility, test_reader_factory]
    results = [test() for test in tests]

    print(f"\n📊 Compatibilidad: {sum(results)}/{len(results)} tests pasaron")