# verify_imports.py

"""

Script para verificar la estructura de imports del módulo null_analysis

y diagnosticar problemas con los tests.



Ejecutar desde la raíz del proyecto:

python verify_imports.py

"""


import os
import sys
from pathlib import Path


def check_imports():
    """Verificar qué se puede importar del módulo null_analysis"""

    print("=" * 60)

    print("VERIFICACIÓN DE IMPORTS - null_analysis")

    print("=" * 60)

    # 1. Verificar estructura de archivos

    print("\n1. ESTRUCTURA DE ARCHIVOS:")

    print("-" * 40)

    base_path = Path("null_analysis")

    if base_path.exists():
        for item in base_path.iterdir():
            if item.is_file() and item.suffix == ".py":
                print(f"  📄 {item.name}")

            elif item.is_dir() and not item.name.startswith("__"):
                print(f"  📁 {item.name}/")

                for subitem in item.iterdir():
                    if subitem.suffix == ".py":
                        print(f"      📄 {subitem.name}")

    else:
        print("  ❌ Directorio null_analysis no encontrado")

    # 2. Verificar imports principales

    print("\n2. VERIFICACIÓN DE IMPORTS PRINCIPALES:")

    print("-" * 40)

    try:
        from null_analysis import NullAnalysisError

        print("  ✅ from null_analysis import NullAnalysisError")

    except ImportError as e:
        print(f"  ❌ from null_analysis import NullAnalysisError - Error: {e}")

    try:
        from null_analysis.exceptions import NullAnalysisError

        print("  ✅ from null_analysis.exceptions import NullAnalysisError")

    except ImportError as e:
        print(f"  ❌ from null_analysis.exceptions import NullAnalysisError - Error: {e}")

    try:
        from null_analysis import InputValidator

        print("  ✅ from null_analysis import InputValidator")

    except ImportError as e:
        print(f"  ❌ from null_analysis import InputValidator - Error: {e}")

    try:
        from null_analysis.utils import InputValidator

        print("  ✅ from null_analysis.utils import InputValidator")

    except ImportError as e:
        print(f"  ❌ from null_analysis.utils import InputValidator - Error: {e}")

    try:
        from null_analysis.utils.utils import InputValidator

        print("  ✅ from null_analysis.utils.utils import InputValidator")

    except ImportError as e:
        print(f"  ❌ from null_analysis.utils.utils import InputValidator - Error: {e}")

    try:
        from null_analysis import safe_dict_merge

        print("  ✅ from null_analysis import safe_dict_merge")

    except ImportError as e:
        print(f"  ❌ from null_analysis import safe_dict_merge - Error: {e}")

    # 3. Verificar qué hay en __init__.py principal

    print("\n3. CONTENIDO DE null_analysis/__init__.py:")

    print("-" * 40)

    init_file = Path("null_analysis/__init__.py")

    if init_file.exists():
        with open(init_file, "r", encoding="utf-8") as f:
            lines = f.readlines()

        # Buscar __all__

        in_all = False

        for line in lines:
            if "__all__" in line:
                in_all = True

            if in_all:
                print(f"  {line.strip()}")

                if "]" in line:
                    break

    else:
        print("  ❌ Archivo __init__.py no encontrado")

    # 4. Verificar ubicación de utils

    print("\n4. UBICACIÓN DE UTILS:")

    print("-" * 40)

    utils_file = Path("null_analysis/utils.py")

    utils_dir = Path("null_analysis/utils")

    if utils_file.exists():
        print("  ✅ utils.py está en null_analysis/utils.py")

        print("     -> Usar: from null_analysis.utils import ...")

    if utils_dir.exists() and utils_dir.is_dir():
        print("  ✅ utils/ es un directorio")

        utils_init = utils_dir / "__init__.py"

        utils_utils = utils_dir / "utils.py"

        if utils_init.exists():
            print("     📄 utils/__init__.py existe")

        if utils_utils.exists():
            print("     📄 utils/utils.py existe")

            print("     -> Usar: from null_analysis.utils.utils import ...")

    # 5. Solución recomendada

    print("\n5. SOLUCIÓN RECOMENDADA PARA LOS TESTS:")

    print("-" * 40)

    print(
        """

Para corregir los tests, usa estos imports en test_suite.py:



```python

# En la parte superior del archivo:

from null_analysis.exceptions import NullAnalysisError



# O si está exportado en __init__.py:

from null_analysis import NullAnalysisError



# En los tests de TestUnitUtils:

def test_input_validator_none_dataframe(self):

    validator = InputValidator()



    with pytest.raises(NullAnalysisError) as exc_info:

        validator.validate_dataframe(None)



    assert "no puede ser None" in str(exc_info.value)

```

    """
    )

    print("\n" + "=" * 60)

    print("VERIFICACIÓN COMPLETA")

    print("=" * 60)


def test_actual_imports():
    """Probar los imports que funcionan realmente"""

    print("\n6. PRUEBA DE IMPORTS FUNCIONALES:")

    print("-" * 40)

    working_imports = []

    failed_imports = []

    # Lista de imports a probar

    imports_to_test = [
        "from null_analysis import NullAnalysisError",
        "from null_analysis.exceptions import NullAnalysisError",
        "from null_analysis import InputValidator",
        "from null_analysis.utils import InputValidator",
        "from null_analysis import safe_dict_merge",
        "from null_analysis.utils import safe_dict_merge",
    ]

    for import_str in imports_to_test:
        try:
            exec(import_str)

            working_imports.append(import_str)

        except ImportError:
            failed_imports.append(import_str)

    print("\n✅ IMPORTS QUE FUNCIONAN:")

    for imp in working_imports:
        print(f"   {imp}")

    if failed_imports:
        print("\n❌ IMPORTS QUE FALLAN:")

        for imp in failed_imports:
            print(f"   {imp}")

    # Sugerir el import correcto

    if working_imports:
        print("\n💡 USA ESTE IMPORT EN TUS TESTS:")

        print(f"   {working_imports[0]}")


if __name__ == "__main__":
    # Agregar el directorio actual al path

    sys.path.insert(0, os.getcwd())

    check_imports()

    test_actual_imports()

    print("\n✨ Para corregir los tests, actualiza los imports según lo encontrado arriba.")

    print("   Los tests están fallando porque intentan importar desde el lugar incorrecto.")

    print("   Usa los imports que están marcados con ✅")
