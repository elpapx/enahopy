#!/usr/bin/env python3
"""
Script de automatización para build y upload a PyPI
===================================================

Script completo para preparar, construir y subir el paquete enaho-analyzer a PyPI.
Incluye verificaciones de calidad, tests, y proceso de release automatizado.

Uso:
    python scripts/build_and_upload.py --test    # Subir a TestPyPI
    python scripts/build_and_upload.py --prod    # Subir a PyPI
    python scripts/build_and_upload.py --check   # Solo verificaciones
"""

import argparse
import subprocess
import sys
import os
import shutil
from pathlib import Path
import re
import json
from typing import List, Dict, Any


class ENAHOPackageBuilder:
    """Automatiza el proceso de build y upload del paquete ENAHO Analyzer."""

    def __init__(self, project_root: Path = None):
        self.project_root = project_root or Path.cwd()
        self.dist_dir = self.project_root / "dist"
        self.build_dir = self.project_root / "build"

    def run_command(self, command: List[str], description: str) -> bool:
        """Ejecuta un comando y maneja errores."""
        print(f"\n🔄 {description}")
        print(f"   Ejecutando: {' '.join(command)}")

        try:
            result = subprocess.run(
                command,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                check=True
            )

            if result.stdout:
                print(f"   ✅ {result.stdout.strip()}")
            return True

        except subprocess.CalledProcessError as e:
            print(f"   ❌ Error: {e}")
            if e.stdout:
                print(f"   Stdout: {e.stdout}")
            if e.stderr:
                print(f"   Stderr: {e.stderr}")
            return False

    def check_environment(self) -> bool:
        """Verifica que el entorno esté configurado correctamente."""
        print("🔍 Verificando entorno de desarrollo...")

        # Verificar Python
        python_version = sys.version_info
        if python_version < (3, 8):
            print(f"❌ Python {python_version} no soportado. Requiere Python 3.8+")
            return False
        print(f"✅ Python {python_version.major}.{python_version.minor}")

        # Verificar herramientas necesarias
        tools = ['build', 'twine', 'pytest', 'black', 'flake8', 'isort']
        missing_tools = []

        for tool in tools:
            try:
                subprocess.run([tool, '--version'], capture_output=True, check=True)
                print(f"✅ {tool}")
            except (subprocess.CalledProcessError, FileNotFoundError):
                missing_tools.append(tool)
                print(f"❌ {tool}")

        if missing_tools:
            print(f"\n📦 Instalar herramientas faltantes:")
            print(f"   pip install {' '.join(missing_tools)}")
            return False

        # Verificar archivos esenciales
        essential_files = [
            'pyproject.toml',
            'README.md',
            'LICENSE',
            '__init__.py'
        ]

        for file in essential_files:
            if not (self.project_root / file).exists():
                print(f"❌ Archivo faltante: {file}")
                return False
            print(f"✅ {file}")

        return True

    def run_tests(self) -> bool:
        """Ejecuta la suite completa de tests."""
        print("\n🧪 Ejecutando tests...")

        # Tests unitarios
        if not self.run_command(['pytest', '-v', '--tb=short'], "Tests unitarios"):
            return False

        # Tests con cobertura
        if not self.run_command(
                ['pytest', '--cov=enaho_analyzer', '--cov-report=term-missing'],
                "Tests con cobertura"
        ):
            return False

        return True

    def check_code_quality(self) -> bool:
        """Verifica calidad del código."""
        print("\n🔍 Verificando calidad del código...")

        # Black formatting
        if not self.run_command(['black', '--check', '.'], "Verificar formateo Black"):
            print("   💡 Ejecuta 'black .' para corregir formato")
            return False

        # Import sorting
        if not self.run_command(['isort', '--check-only', '.'], "Verificar orden de imports"):
            print("   💡 Ejecuta 'isort .' para corregir imports")
            return False

        # Flake8 linting
        if not self.run_command(['flake8', '.'], "Verificar linting"):
            return False

        return True

    def update_version_if_needed(self) -> str:
        """Actualiza versión si es necesario y retorna la versión actual."""
        pyproject_path = self.project_root / "pyproject.toml"

        with open(pyproject_path, 'r') as f:
            content = f.read()

        # Extraer versión actual
        version_match = re.search(r'version = "([^"]+)"', content)
        if not version_match:
            raise ValueError("No se pudo encontrar versión en pyproject.toml")

        current_version = version_match.group(1)
        print(f"📋 Versión actual: {current_version}")

        # Aquí podrías implementar lógica de auto-incremento
        # Por ahora, solo retornamos la versión actual
        return current_version

    def clean_build_artifacts(self) -> bool:
        """Limpia artifacts de builds anteriores."""
        print("\n🧹 Limpiando artifacts anteriores...")

        dirs_to_clean = [self.dist_dir, self.build_dir]
        dirs_to_clean.extend(self.project_root.glob("*.egg-info"))

        for dir_path in dirs_to_clean:
            if dir_path.exists():
                shutil.rmtree(dir_path)
                print(f"   🗑️  Eliminado: {dir_path}")

        return True

    def build_package(self) -> bool:
        """Construye el paquete."""
        print("\n🔨 Construyendo paquete...")

        # Build con python -m build
        if not self.run_command(
                [sys.executable, '-m', 'build'],
                "Construir distribuciones"
        ):
            return False

        # Verificar que se crearon los archivos
        if not self.dist_dir.exists():
            print("❌ No se creó directorio dist/")
            return False

        dist_files = list(self.dist_dir.glob("*"))
        if len(dist_files) < 2:  # Esperamos .tar.gz y .whl
            print(f"❌ Solo se crearon {len(dist_files)} archivos de distribución")
            return False

        print(f"✅ Archivos creados:")
        for file in dist_files:
            print(f"   📦 {file.name}")

        return True

    def check_package(self) -> bool:
        """Verifica el paquete construido."""
        print("\n🔍 Verificando paquete...")

        # Verificar con twine
        return self.run_command(
            ['twine', 'check', 'dist/*'],
            "Verificar paquete con twine"
        )

    def upload_to_test_pypi(self) -> bool:
        """Sube el paquete a TestPyPI."""
        print("\n🚀 Subiendo a TestPyPI...")

        return self.run_command(
            ['twine', 'upload', '--repository', 'testpypi', 'dist/*'],
            "Upload a TestPyPI"
        )

    def upload_to_pypi(self) -> bool:
        """Sube el paquete a PyPI."""
        print("\n🚀 Subiendo a PyPI...")

        # Confirmación adicional
        confirm = input("¿Estás seguro de subir a PyPI PRODUCCIÓN? (yes/no): ")
        if confirm.lower() != 'yes':
            print("❌ Upload cancelado")
            return False

        return self.run_command(
            ['twine', 'upload', 'dist/*'],
            "Upload a PyPI"
        )

    def test_installation(self, test_pypi: bool = False) -> bool:
        """Prueba la instalación del paquete."""
        print(f"\n🧪 Probando instalación desde {'TestPyPI' if test_pypi else 'PyPI'}...")

        # Crear entorno temporal
        temp_env = self.project_root / "temp_test_env"

        try:
            # Crear entorno virtual temporal
            subprocess.run([sys.executable, '-m', 'venv', str(temp_env)], check=True)

            # Determinar comando pip en el entorno
            if sys.platform == "win32":
                pip_cmd = temp_env / "Scripts" / "pip"
            else:
                pip_cmd = temp_env / "bin" / "pip"

            # Instalar paquete
            if test_pypi:
                install_cmd = [
                    str(pip_cmd), 'install',
                    '--index-url', 'https://test.pypi.org/simple/',
                    '--extra-index-url', 'https://pypi.org/simple/',
                    'enaho-analyzer'
                ]
            else:
                install_cmd = [str(pip_cmd), 'install', 'enaho-analyzer']

            if not self.run_command(install_cmd, "Instalar paquete"):
                return False

            # Probar import básico
            python_cmd = temp_env / ("Scripts/python" if sys.platform == "win32" else "bin/python")
            test_cmd = [
                str(python_cmd), '-c',
                'import enaho_analyzer; print(f"✅ enaho_analyzer {enaho_analyzer.__version__} instalado correctamente")'
            ]

            return self.run_command(test_cmd, "Probar import")

        except Exception as e:
            print(f"❌ Error en test de instalación: {e}")
            return False
        finally:
            # Limpiar entorno temporal
            if temp_env.exists():
                shutil.rmtree(temp_env)

    def create_release_notes(self, version: str) -> bool:
        """Crea notas de release automáticas."""
        print(f"\n📝 Creando notas de release para v{version}...")

        # Leer CHANGELOG.md
        changelog_path = self.project_root / "CHANGELOG.md"
        if not changelog_path.exists():
            print("❌ CHANGELOG.md no encontrado")
            return False

        with open(changelog_path, 'r', encoding='utf-8') as f:
            changelog_content = f.read()

        # Extraer sección de la versión actual
        version_pattern = rf"## \[{re.escape(version)}\].*?(?=## \[|\Z)"
        version_match = re.search(version_pattern, changelog_content, re.DOTALL)

        if not version_match:
            print(f"❌ No se encontró información para versión {version} en CHANGELOG.md")
            return False

        release_notes = version_match.group(0).strip()

        # Guardar notas de release
        release_notes_path = self.project_root / f"release_notes_v{version}.md"
        with open(release_notes_path, 'w', encoding='utf-8') as f:
            f.write(release_notes)

        print(f"✅ Notas de release guardadas en: {release_notes_path}")
        return True

    def full_release_process(self, target: str = "test") -> bool:
        """Ejecuta el proceso completo de release."""
        print("🚀 ENAHO ANALYZER - PROCESO DE RELEASE")
        print("=" * 50)

        steps = [
            ("Verificar entorno", self.check_environment),
            ("Verificar calidad de código", self.check_code_quality),
            ("Ejecutar tests", self.run_tests),
            ("Limpiar artifacts", self.clean_build_artifacts),
            ("Construir paquete", self.build_package),
            ("Verificar paquete", self.check_package),
        ]

        # Ejecutar pasos previos
        for description, step_func in steps:
            if not step_func():
                print(f"\n❌ FALLÓ: {description}")
                return False

        # Obtener versión
        version = self.update_version_if_needed()

        # Crear notas de release
        self.create_release_notes(version)

        # Upload según target
        if target == "test":
            if not self.upload_to_test_pypi():
                print("\n❌ FALLÓ: Upload a TestPyPI")
                return False

            # Test installation opcional
            test_install = input("\n¿Probar instalación desde TestPyPI? (y/n): ")
            if test_install.lower() in ['y', 'yes']:
                self.test_installation(test_pypi=True)

        elif target == "prod":
            if not self.upload_to_pypi():
                print("\n❌ FALLÓ: Upload a PyPI")
                return False

            print("\n⏳ Esperando propagación en PyPI...")
            import time
            time.sleep(30)  # Esperar propagación

            self.test_installation(test_pypi=False)

        print(f"\n🎉 ¡RELEASE {version} COMPLETADO EXITOSAMENTE!")
        print("\n📋 Próximos pasos:")
        print("   1. Crear release en GitHub con las notas generadas")
        print("   2. Actualizar documentación si es necesario")
        print("   3. Anunciar en redes sociales/comunidades")

        return True


def main():
    """Función principal del script."""
    parser = argparse.ArgumentParser(
        description="Build and upload enaho-analyzer to PyPI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python scripts/build_and_upload.py --check         # Solo verificaciones
  python scripts/build_and_upload.py --test          # Upload a TestPyPI  
  python scripts/build_and_upload.py --prod          # Upload a PyPI
  python scripts/build_and_upload.py --build-only    # Solo build local
        """
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--check', action='store_true',
                       help='Solo ejecutar verificaciones (tests, calidad)')
    group.add_argument('--test', action='store_true',
                       help='Build y upload a TestPyPI')
    group.add_argument('--prod', action='store_true',
                       help='Build y upload a PyPI (PRODUCCIÓN)')
    group.add_argument('--build-only', action='store_true',
                       help='Solo build local sin upload')

    parser.add_argument('--project-root', type=Path,
                        default=Path.cwd(),
                        help='Directorio raíz del proyecto')

    args = parser.parse_args()

    # Crear builder
    builder = ENAHOPackageBuilder(args.project_root)

    try:
        if args.check:
            print("🔍 MODO VERIFICACIÓN - Solo checks de calidad")
            success = (
                    builder.check_environment() and
                    builder.check_code_quality() and
                    builder.run_tests()
            )

        elif args.build_only:
            print("🔨 MODO BUILD - Solo construcción local")
            success = (
                    builder.check_environment() and
                    builder.check_code_quality() and
                    builder.run_tests() and
                    builder.clean_build_artifacts() and
                    builder.build_package() and
                    builder.check_package()
            )

        elif args.test:
            print("🧪 MODO TEST - Upload a TestPyPI")
            success = builder.full_release_process("test")

        elif args.prod:
            print("🚀 MODO PRODUCCIÓN - Upload a PyPI")
            success = builder.full_release_process("prod")

        if success:
            print("\n✅ Proceso completado exitosamente")
            sys.exit(0)
        else:
            print("\n❌ Proceso falló")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n\n⚠️ Proceso cancelado por usuario")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error inesperado: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()