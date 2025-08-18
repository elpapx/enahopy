# test_structure.py (en tu directorio raíz)

def test_imports():
    """Prueba que la nueva estructura se puede importar"""
    try:
        print("🔍 Probando importaciones...")

        # 1. Importar desde la nueva estructura core
        from enahopy.null_analysis.core.config import NullAnalysisConfig
        from enahopy.null_analysis.core.exceptions import NullAnalysisError

        # Crear configuración de prueba
        config = NullAnalysisConfig()
        print(f"✅ Nueva configuración creada: {config.complexity_level}")

        # 2. Importar desde el __init__.py del módulo null_analysis
        from enahopy.null_analysis import (
            NullAnalysisConfig as ConfigFromInit,
            MissingDataPattern,
            AnalysisComplexity
        )
        print("✅ Importación desde __init__.py del módulo funciona")

        # 3. Probar que las configuraciones son consistentes
        config2 = ConfigFromInit()
        assert config.complexity_level == config2.complexity_level
        print("✅ Configuraciones son consistentes")

        # 4. Intentar importar API legacy (puede fallar y está bien)
        try:
            from enahopy.null_analysis import ENAHONullAnalyzer, quick_null_analysis
            print("✅ API legacy también disponible")
            legacy_available = True
        except ImportError:
            print("⚠️  API legacy no disponible aún (esperado)")
            legacy_available = False

        # 5. Verificar enums
        pattern = MissingDataPattern.MCAR
        complexity = AnalysisComplexity.STANDARD
        print(f"✅ Enums funcionando: {pattern.value}, {complexity.value}")

        print("\n🎉 Estructura base creada exitosamente!")
        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_directory_structure():
    """Verifica que la estructura de directorios sea correcta"""
    import os
    from pathlib import Path

    print("\n🔍 Verificando estructura de directorios...")

    # Directorio base del proyecto
    base_dir = Path("enahopy")

    expected_files = [
        "enahopy/__init__.py",
        "enahopy/loader.py",
        "enahopy/merger.py",
        "enahopy/null_values_analyzer.py",  # Archivo original
        "enahopy/null_analysis/__init__.py",
        "enahopy/null_analysis/core/__init__.py",
        "enahopy/null_analysis/core/config.py",
        "enahopy/null_analysis/core/exceptions.py"
    ]

    missing_files = []
    for file_path in expected_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
        else:
            print(f"✅ {file_path}")

    if missing_files:
        print(f"\n❌ Archivos faltantes:")
        for file_path in missing_files:
            print(f"   - {file_path}")
        return False
    else:
        print("\n✅ Todos los archivos necesarios están presentes")
        return True


if __name__ == "__main__":
    print("=" * 50)
    print("PRUEBA DE ESTRUCTURA MODULAR")
    print("=" * 50)

    # Verificar estructura de archivos
    structure_ok = test_directory_structure()

    if structure_ok:
        # Verificar importaciones
        import_ok = test_imports()

        if import_ok:
            print("\n🏆 ¡TODO FUNCIONANDO CORRECTAMENTE!")
        else:
            print("\n💥 Problemas con las importaciones")
    else:
        print("\n💥 Problemas con la estructura de directorios")