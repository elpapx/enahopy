"""
Script principal para ejecutar todos los tests
"""


def run_all_tests():
    """Ejecuta todos los tests de la refactorización"""
    print("🚀 TESTING COMPLETO DE REFACTORIZACIÓN")
    print("=" * 50)

    # Lista de módulos de test
    test_modules = [
        'test_refactoring',
        'test_compatibility',
        'test_real_file'
    ]

    results = {}

    for module_name in test_modules:
        print(f"\n📋 Ejecutando {module_name}...")
        print("-" * 30)

        try:
            # Importar dinámicamente
            module = __import__(module_name)

            # Ejecutar tests del módulo
            if hasattr(module, '__main__') or hasattr(module, 'main'):
                exec(open(f"{module_name}.py").read())
                results[module_name] = "✅ PASSED"
            else:
                results[module_name] = "⚠️  NO MAIN"

        except ImportError as e:
            results[module_name] = f"❌ IMPORT ERROR: {e}"
        except Exception as e:
            results[module_name] = f"❌ ERROR: {e}"

    # Resumen final
    print("\n" + "=" * 50)
    print("📊 RESUMEN DE TESTS")
    print("=" * 50)

    for module, result in results.items():
        print(f"{module:20} : {result}")

    passed = sum(1 for r in results.values() if "✅" in r)
    total = len(results)

    print(f"\n🎯 RESULTADO FINAL: {passed}/{total} módulos OK")

    if passed == total:
        print("🎉 ¡REFACTORIZACIÓN EXITOSA!")
        print("💡 Todo está funcionando correctamente.")
    else:
        print("⚠️  Revisar módulos con errores.")

    return passed == total


if __name__ == "__main__":
    run_all_tests()