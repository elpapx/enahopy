#!/usr/bin/env python3
"""
Verificación Inmediata del Estado Actual
========================================

Script para verificar exactamente qué tenemos en los archivos clave
antes de proceder con la finalización de la refactorización.
"""

import os
import sys
from pathlib import Path


def show_file_content(filepath, max_lines=50):
    """Muestra contenido de un archivo con límite de líneas"""
    try:
        path = Path(filepath)
        if not path.exists():
            print(f"❌ No existe: {filepath}")
            return

        content = path.read_text(encoding="utf-8")
        lines = content.split("\n")

        print(f"\n📄 {filepath} ({len(lines)} líneas):")
        print("=" * 60)

        for i, line in enumerate(lines[:max_lines], 1):
            print(f"{i:3}: {line}")

        if len(lines) > max_lines:
            print(f"... y {len(lines) - max_lines} líneas más")

    except Exception as e:
        print(f"❌ Error leyendo {filepath}: {e}")


def test_imports():
    """Prueba diferentes formas de importar"""
    print(f"\n🧪 PROBANDO IMPORTS DETALLADAMENTE")
    print("=" * 60)

    # Añadir path actual al sys.path
    current_dir = Path.cwd()
    if str(current_dir) not in sys.path:
        sys.path.insert(0, str(current_dir))

    import_tests = [
        # Imports de módulos
        ("import enahopy", "módulo enahopy"),
        ("import enahopy.merger", "módulo merger"),
        ("from enahopy import merger", "merger desde enahopy"),
        # Imports de clases desde __init__.py
        ("from enahopy.merger import ENAHOGeoMerger", "ENAHOGeoMerger desde __init__"),
        ("from enahopy.merger import ENAHOModuleMerger", "ENAHOModuleMerger desde __init__"),
        ("from enahopy.merger import merge_enaho_modules", "función merge_enaho_modules"),
        ("from enahopy.merger import GeoMergeConfiguration", "GeoMergeConfiguration"),
        # Imports directos desde archivos
        ("from enahopy.merger.core import ENAHOGeoMerger", "ENAHOGeoMerger desde core"),
        (
            "from enahopy.merger.modules.merger import ENAHOModuleMerger",
            "ENAHOModuleMerger directo",
        ),
        (
            "from enahopy.merger.geographic.merger import GeographicMerger",
            "GeographicMerger directo",
        ),
    ]

    for import_stmt, description in import_tests:
        try:
            exec(import_stmt)
            print(f"✅ {import_stmt}")
        except ImportError as e:
            print(f"❌ {import_stmt}")
            print(f"   💡 Error: {e}")
        except Exception as e:
            print(f"⚠️ {import_stmt}")
            print(f"   💡 Error inesperado: {e}")


def analyze_class_structure():
    """Analiza estructura de clases disponibles"""
    print(f"\n🔍 ANALIZANDO ESTRUCTURA DE CLASES")
    print("=" * 60)

    try:
        # Importar módulo merger
        import enahopy.merger as merger

        # Listar atributos públicos
        public_attrs = [attr for attr in dir(merger) if not attr.startswith("_")]
        print(f"📋 Atributos públicos en enahopy.merger ({len(public_attrs)}):")

        for attr in public_attrs[:20]:  # Primeros 20
            obj = getattr(merger, attr)
            obj_type = type(obj).__name__
            print(f"  • {attr} ({obj_type})")

        if len(public_attrs) > 20:
            print(f"  ... y {len(public_attrs) - 20} más")

        # Probar instanciación de clases clave
        print(f"\n🏗️ PROBANDO INSTANCIACIÓN:")

        if hasattr(merger, "ENAHOGeoMerger"):
            geo_merger = merger.ENAHOGeoMerger()
            print(f"✅ ENAHOGeoMerger instanciado")
            print(f"   📋 Métodos: {len([m for m in dir(geo_merger) if not m.startswith('_')])}")

        if hasattr(merger, "ENAHOModuleMerger"):
            # Necesita configuración
            from enahopy.merger import ModuleMergeConfig

            config = ModuleMergeConfig()
            module_merger = merger.ENAHOModuleMerger(config)
            print(f"✅ ENAHOModuleMerger instanciado")
            print(f"   📋 Métodos: {len([m for m in dir(module_merger) if not m.startswith('_')])}")

    except Exception as e:
        print(f"❌ Error en análisis de estructura: {e}")
        import traceback

        traceback.print_exc()


def main():
    """Ejecuta verificación completa"""
    print("🚀 VERIFICACIÓN INMEDIATA DEL ESTADO ACTUAL")
    print("=" * 60)

    # 1. Mostrar contenido de archivos clave
    key_files = [
        "enahopy/merger/geographic/merger.py",
        "enahopy/merger/geographic/__init__.py",
        "enahopy/merger/modules/__init__.py",
    ]

    print("📁 CONTENIDO DE ARCHIVOS CLAVE:")
    for filepath in key_files:
        show_file_content(filepath, max_lines=30)

    # 2. Probar imports
    test_imports()

    # 3. Analizar estructura
    analyze_class_structure()

    print(f"\n🎯 CONCLUSIONES:")
    print("=" * 60)
    print("1. ✅ Si los imports funcionan → API está disponible")
    print("2. ❌ Si GeographicMerger es básico → Necesita completarse")
    print("3. ✅ Si ENAHOModuleMerger funciona → Componente listo")
    print("4. 🎯 Próximo paso: Completar componente faltante")


if __name__ == "__main__":
    main()
