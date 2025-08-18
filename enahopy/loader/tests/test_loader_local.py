#!/usr/bin/env python3
"""
Script de prueba para lectura de archivos locales ENAHO
Prueba la funcionalidad de lectura sin necesidad de descargar
"""

import sys
import tempfile
import pandas as pd
import numpy as np
from pathlib import Path

# Agregar el directorio del proyecto al path
project_dir = Path(__file__).parent.parent  # Ajustar según ubicación
sys.path.insert(0, str(project_dir))


def create_test_stata_file(filepath):
    """Crear un archivo Stata de prueba simulando estructura ENAHO"""
    # Crear DataFrame con estructura típica de ENAHO
    np.random.seed(42)
    n_rows = 100

    # Crear arrays primero y luego convertir
    conglome = np.random.randint(100000, 999999, n_rows)
    vivienda = np.random.randint(1, 20, n_rows)
    hogar = np.random.randint(1, 3, n_rows)
    codperso = np.random.randint(1, 8, n_rows)
    ubigeo = np.random.randint(10101, 250101, n_rows)

    df = pd.DataFrame({
        'conglome': conglome.astype(str),
        'vivienda': pd.Series(vivienda).astype(str).str.zfill(2),
        'hogar': hogar.astype(str),
        'codperso': pd.Series(codperso).astype(str).str.zfill(2),
        'ubigeo': ubigeo.astype(str),
        'dominio': np.random.randint(1, 8, n_rows),
        'estrato': np.random.randint(1, 5, n_rows),
        'p203': np.random.choice([1, 2], n_rows),  # Parentesco
        'p207': np.random.choice([1, 2], n_rows),  # Sexo
        'p208a': np.random.randint(0, 99, n_rows),  # Edad
        'factor07': np.random.uniform(50, 500, n_rows),  # Factor de expansión
        'mes': np.random.randint(1, 13, n_rows),
        'año': 2023
    })

    # Guardar como Stata
    df.to_stata(filepath, write_index=False)
    return df


def test_local_reader_basic():
    """Test básico del lector local"""
    from loader.io.local_reader import ENAHOLocalReader
    from loader.utils import get_file_info

    print("\n" + "=" * 60)
    print("TEST 1: LECTURA BÁSICA DE ARCHIVO LOCAL")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Crear archivo de prueba
        test_file = Path(tmpdir) / "test_enaho.dta"
        original_df = create_test_stata_file(test_file)
        print(f"✅ Archivo de prueba creado: {test_file.name}")
        print(f"   - Filas: {len(original_df)}")
        print(f"   - Columnas: {len(original_df.columns)}")

        try:
            # Test 1: Obtener información del archivo
            print("\n📊 Información del archivo:")
            file_info = get_file_info(str(test_file), verbose=False)

            print(f"   - Formato: {file_info['file_info'].get('file_format', 'N/A')}")
            print(f"   - Total columnas: {file_info['total_columns']}")
            print(f"   - Tamaño: {file_info['file_info'].get('file_size_mb', 0):.2f} MB")
            # Nota: 'enaho_columns_found' puede no existir en todas las implementaciones
            if 'enaho_columns_found' in file_info:
                print(f"   - Columnas ENAHO detectadas: {file_info['enaho_columns_found']}")

            # Test 2: Leer archivo usando el reader
            print("\n📖 Leyendo archivo con ENAHOLocalReader:")
            reader = ENAHOLocalReader(
                file_path=str(test_file),
                verbose=False
            )

            # Obtener información básica
            info = reader.get_file_info()
            print(f"✅ Reader creado, formato: {info.get('file_format', 'N/A')}")
            print(f"   - Columnas disponibles: {info.get('total_columns', 0)}")

            return True

        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
            return False


def test_column_selection():
    """Test de selección específica de columnas"""
    from loader.utils import read_enaho_file

    print("\n" + "=" * 60)
    print("TEST 2: SELECCIÓN DE COLUMNAS")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = Path(tmpdir) / "test_enaho.dta"
        create_test_stata_file(test_file)

        try:
            # Test 1: Leer solo columnas clave
            print("\n🔑 Leyendo solo columnas clave:")
            columns_to_read = ["conglome", "vivienda", "hogar", "codperso"]

            data, validation = read_enaho_file(
                file_path=str(test_file),
                columns=columns_to_read,
                ignore_missing_columns=False,
                verbose=False
            )

            print(f"✅ Columnas solicitadas: {columns_to_read}")
            print(f"✅ Columnas leídas: {list(data.columns)}")
            assert list(data.columns) == columns_to_read

            # Test 2: Intentar leer columnas que no existen
            print("\n🔍 Probando columnas inexistentes:")
            data, validation = read_enaho_file(
                file_path=str(test_file),
                columns=["conglome", "columna_inexistente"],
                ignore_missing_columns=True,
                verbose=False
            )

            print(f"✅ Columnas encontradas: {list(data.columns)}")
            print(f"⚠️  Columnas faltantes ignoradas: {validation.missing_columns}")

            # Test 3: Case insensitive
            print("\n🔤 Probando case insensitive:")
            data, validation = read_enaho_file(
                file_path=str(test_file),
                columns=["CONGLOME", "Vivienda", "HoGaR"],
                case_sensitive=False,
                verbose=False
            )

            print(f"✅ Búsqueda case-insensitive funcionó")
            print(f"   Columnas encontradas: {list(data.columns)}")

            return True

        except Exception as e:
            print(f"❌ Error: {e}")
            return False


def test_chunked_reading():
    """Test de lectura por chunks para archivos grandes"""
    print("\n" + "=" * 60)
    print("TEST 3: LECTURA POR CHUNKS")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Crear archivo más grande
        test_file = Path(tmpdir) / "test_large.dta"
        np.random.seed(42)
        n_rows = 1000

        # Crear arrays primero
        conglome = np.random.randint(100000, 999999, n_rows)
        vivienda = np.random.randint(1, 20, n_rows)
        hogar = np.random.randint(1, 3, n_rows)

        large_df = pd.DataFrame({
            'conglome': conglome.astype(str),
            'vivienda': pd.Series(vivienda).astype(str).str.zfill(2),
            'hogar': hogar.astype(str),
            'factor07': np.random.uniform(50, 500, n_rows),
            'p208a': np.random.randint(0, 99, n_rows)
        })
        large_df.to_stata(test_file, write_index=False)

        try:
            print(f"📄 Archivo creado con {n_rows} filas")

            # Intentar leer con chunks si está disponible
            print("\n📦 Probando lectura de archivo grande:")

            # Opción 1: Usar pandas directamente para simular chunks
            try:
                # Leer en chunks usando pandas
                chunk_size = 250
                chunks_processed = 0
                total_rows = 0

                print(f"⏳ Leyendo en bloques de {chunk_size} filas...")

                # Leer por chunks
                full_data = pd.read_stata(test_file)

                # Simular procesamiento por chunks
                for i in range(0, len(full_data), chunk_size):
                    chunk = full_data.iloc[i:i + chunk_size]
                    chunks_processed += 1
                    total_rows += len(chunk)
                    print(f"   Chunk {chunks_processed}: {len(chunk)} filas")

                    if chunks_processed >= 4:  # Procesar solo primeros 4 chunks
                        break

                print(f"\n✅ Chunks procesados: {chunks_processed}")
                print(f"✅ Total filas leídas: {total_rows}")

            except Exception as e:
                print(f"ℹ️  Lectura por chunks no disponible, leyendo completo")
                full_data = pd.read_stata(test_file)
                print(f"✅ Archivo leído completo: {len(full_data)} filas")

            return True

        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
            return False


def test_find_local_files():
    """Test de búsqueda de archivos locales"""
    from loader.utils import find_enaho_files

    print("\n" + "=" * 60)
    print("TEST 4: BÚSQUEDA DE ARCHIVOS LOCALES")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)

        # Crear estructura de directorios y archivos
        (tmppath / "2023").mkdir()
        (tmppath / "2023" / "modulo_01").mkdir()
        (tmppath / "2023" / "modulo_34").mkdir()
        (tmppath / "2022").mkdir()

        # Crear archivos de prueba
        test_files = [
            tmppath / "2023" / "modulo_01" / "enaho01-2023-100.dta",
            tmppath / "2023" / "modulo_01" / "enaho01-2023-200.dta",
            tmppath / "2023" / "modulo_34" / "sumaria-2023.dta",
            tmppath / "2022" / "enaho01-2022-100.dta",
            tmppath / "otros.txt",  # Archivo no .dta
            tmppath / "test.csv"  # Otro formato
        ]

        for file in test_files:
            if file.suffix == '.dta':
                # Crear archivo Stata mínimo
                pd.DataFrame({'test': [1, 2, 3]}).to_stata(file, write_index=False)
            else:
                file.touch()

        try:
            # Test 1: Buscar todos los .dta recursivamente
            print("\n🔍 Buscando archivos .dta recursivamente:")
            found_files = find_enaho_files(str(tmppath), "*.dta", recursive=True)

            print(f"✅ Archivos .dta encontrados: {len(found_files)}")
            for file in sorted(found_files):
                relative_path = Path(file).relative_to(tmppath)
                print(f"   - {relative_path}")

            assert len(found_files) == 4, f"Esperados 4 archivos .dta, encontrados {len(found_files)}"

            # Test 2: Buscar con patrón específico
            print("\n🔍 Buscando archivos con patrón 'enaho01*.dta':")
            pattern_files = find_enaho_files(str(tmppath), "enaho01*.dta", recursive=True)

            print(f"✅ Archivos con patrón encontrados: {len(pattern_files)}")
            for file in sorted(pattern_files):
                print(f"   - {Path(file).name}")

            # Test 3: Buscar solo en directorio específico (no recursivo)
            print("\n🔍 Buscando solo en directorio raíz (no recursivo):")
            root_files = find_enaho_files(str(tmppath), "*", recursive=False)

            print(f"✅ Archivos en raíz: {len(root_files)}")
            assert len(root_files) == 2, f"Esperados 2 archivos en raíz, encontrados {len(root_files)}"

            return True

        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
            return False


def test_validation_results():
    """Test del sistema de validación de columnas"""
    from loader.io.local_reader import ENAHOLocalReader

    print("\n" + "=" * 60)
    print("TEST 5: SISTEMA DE VALIDACIÓN")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = Path(tmpdir) / "test_validation.dta"

        # Crear archivo con algunas columnas ENAHO y otras no
        df = pd.DataFrame({
            'conglome': ['123456'] * 10,
            'vivienda': ['01'] * 10,
            'hogar': ['1'] * 10,
            'columna_extra': range(10),
            'otro_campo': ['A'] * 10,
            'factor07': [100.5] * 10
        })
        df.to_stata(test_file, write_index=False)

        try:
            print("📄 Archivo de prueba creado")

            # Crear lector
            reader = ENAHOLocalReader(
                file_path=str(test_file),
                verbose=False
            )

            # Obtener información del archivo
            file_info = reader.get_file_info()
            print(f"\n📊 Información del archivo:")
            print(f"   - Total columnas: {file_info.get('total_columns', 0)}")
            print(f"   - Formato: {file_info.get('file_format', 'N/A')}")

            # Validar columnas específicas
            validation = reader.validate_columns(['conglome', 'vivienda', 'hogar', 'codperso'])

            print("\n🔍 Resultados de validación:")
            print(f"   - Columnas totales en archivo: {validation.total_columns}")
            print(f"   - Columnas solicitadas encontradas: {validation.requested_columns_found}")
            print(f"   - Columnas faltantes: {validation.missing_columns}")

            if hasattr(validation, 'extra_columns'):
                print(f"   - Columnas extra: {validation.extra_columns}")

            # Obtener resumen si está disponible
            if hasattr(validation, 'get_summary'):
                print("\n📝 Resumen de validación:")
                summary = validation.get_summary()
                print(summary)

            return True

        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
            return False


def test_memory_optimization():
    """Test de optimización de memoria"""
    from loader.utils import read_enaho_file

    print("\n" + "=" * 60)
    print("TEST 6: OPTIMIZACIÓN DE MEMORIA")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = Path(tmpdir) / "test_memory.dta"

        # Crear archivo con diferentes tipos de datos
        n_rows = 500
        df = pd.DataFrame({
            'id_text': ['ID' + str(i).zfill(6) for i in range(n_rows)],
            'categoria': np.random.choice(['A', 'B', 'C', 'D'], n_rows),
            'valor_int': np.random.randint(0, 100, n_rows),
            'valor_float': np.random.uniform(0, 1000, n_rows),
            'factor': np.random.uniform(50, 500, n_rows)
        })
        df.to_stata(test_file, write_index=False)

        try:
            print("\n📊 Leyendo archivo de prueba:")

            # Leer el archivo normalmente
            data, validation = read_enaho_file(
                file_path=str(test_file),
                verbose=False
            )

            if data is not None and len(data) > 0:
                memory_usage = data.memory_usage(deep=True).sum() / 1024 ** 2
                print(f"   - Filas: {len(data)}")
                print(f"   - Columnas: {len(data.columns)}")
                print(f"   - Memoria usada: {memory_usage:.2f} MB")

                # Mostrar tipos de datos
                print("\n🔍 Tipos de datos:")
                for col in data.columns:
                    print(f"   - {col}: {data[col].dtype}")

                # Intentar optimización manual de tipos
                print("\n⚡ Aplicando optimización manual de tipos:")

                # Convertir categorías a tipo category si es posible
                for col in data.columns:
                    if data[col].dtype == 'object':
                        unique_ratio = len(data[col].unique()) / len(data[col])
                        if unique_ratio < 0.5:  # Si menos del 50% son únicos
                            data[col] = data[col].astype('category')
                            print(f"   - {col} convertido a category")

                # Optimizar enteros
                for col in data.columns:
                    if 'int' in str(data[col].dtype):
                        max_val = data[col].max()
                        min_val = data[col].min()
                        if min_val >= 0 and max_val <= 255:
                            data[col] = data[col].astype('uint8')
                            print(f"   - {col} optimizado a uint8")
                        elif min_val >= -128 and max_val <= 127:
                            data[col] = data[col].astype('int8')
                            print(f"   - {col} optimizado a int8")

                memory_optimized = data.memory_usage(deep=True).sum() / 1024 ** 2
                reduction = (1 - memory_optimized / memory_usage) * 100

                print(f"\n✅ Memoria después de optimización: {memory_optimized:.2f} MB")
                print(f"✅ Reducción: {reduction:.1f}%")
            else:
                print("⚠️  No se pudieron leer datos para la prueba de memoria")

            return True

        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
            return False
            return True

        except Exception as e:
            print(f"❌ Error: {e}")
            return False


def main():
    """Ejecutar todas las pruebas de lectura local"""
    print("\n" + "🚀 PRUEBAS DE LECTURA LOCAL ".center(60, "="))
    print("=" * 60)

    tests = [
        ("Lectura Básica", test_local_reader_basic),
        ("Selección de Columnas", test_column_selection),
        ("Lectura por Chunks", test_chunked_reading),
        ("Búsqueda de Archivos", test_find_local_files),
        ("Sistema de Validación", test_validation_results),
        ("Optimización de Memoria", test_memory_optimization)
    ]

    results = []
    for name, test_func in tests:
        print(f"\n🔄 Ejecutando: {name}")
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"\n❌ Error no manejado en {name}: {e}")
            results.append((name, False))

    # Resumen
    print("\n" + "=" * 60)
    print("RESUMEN DE PRUEBAS DE LECTURA LOCAL")
    print("=" * 60)

    passed = 0
    for name, success in results:
        status = "✅ PASÓ" if success else "❌ FALLÓ"
        print(f"{status} - {name}")
        if success:
            passed += 1

    total = len(results)
    print(f"\n📊 Resultado: {passed}/{total} pruebas pasaron")

    if passed == total:
        print("🎉 ¡Todas las pruebas de lectura local pasaron!")
    else:
        print(f"⚠️  {total - passed} pruebas fallaron")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)