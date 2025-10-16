"""
Command Line Interface para enahopy
"""

import argparse
import sys
from pathlib import Path

from . import __version__


def main():
    """Función principal del CLI"""
    parser = argparse.ArgumentParser(
        description="enahopy - Herramienta para análisis de datos ENAHO",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  enahopy download --year 2023 --module 01
  enahopy merge --input file1.dta file2.dta --output merged.csv
  enahopy analyze --file data.dta --report nulls
        """,
    )

    parser.add_argument("--version", action="version", version=f"enahopy {__version__}")

    # Subcomandos
    subparsers = parser.add_subparsers(dest="command", help="Comandos disponibles")

    # Comando download
    download_parser = subparsers.add_parser("download", help="Descargar datos ENAHO")
    download_parser.add_argument("--year", type=str, required=True, help="Año de la encuesta")
    download_parser.add_argument("--module", type=str, required=True, help="Código del módulo")
    download_parser.add_argument("--output", type=str, help="Directorio de salida")

    # Comando merge
    merge_parser = subparsers.add_parser("merge", help="Fusionar módulos ENAHO")
    merge_parser.add_argument("--input", nargs="+", required=True, help="Archivos a fusionar")
    merge_parser.add_argument("--output", type=str, required=True, help="Archivo de salida")
    merge_parser.add_argument("--level", choices=["hogar", "individuo"], default="hogar")

    # Comando analyze
    analyze_parser = subparsers.add_parser("analyze", help="Analizar datos ENAHO")
    analyze_parser.add_argument("--file", type=str, required=True, help="Archivo a analizar")
    analyze_parser.add_argument(
        "--report", choices=["nulls", "quality", "summary"], default="summary"
    )
    analyze_parser.add_argument("--output", type=str, help="Archivo de reporte")

    args = parser.parse_args()

    # Ejecutar comando correspondiente
    if args.command == "download":
        return download_command(args)
    elif args.command == "merge":
        return merge_command(args)
    elif args.command == "analyze":
        return analyze_command(args)
    else:
        parser.print_help()
        return 0


def download_command(args):
    """Ejecutar comando de descarga"""
    try:
        from .loader import download_enaho_data

        print(f"Descargando módulo {args.module} del año {args.year}...")
        df = download_enaho_data(
            modulos=[args.module], años=[args.year], directorio_salida=args.output
        )
        print(f"Descarga completada: {len(df)} registros")
        return 0
    except Exception as e:
        print(f"Error en descarga: {e}", file=sys.stderr)
        return 1


def merge_command(args):
    """Ejecutar comando de fusión"""
    try:
        import pandas as pd

        from .merger import merge_enaho_modules

        print(f"Fusionando {len(args.input)} archivos...")

        # Leer archivos
        dataframes = {}
        for i, filepath in enumerate(args.input):
            df = pd.read_stata(filepath) if filepath.endswith(".dta") else pd.read_csv(filepath)
            dataframes[f"modulo_{i}"] = df

        # Fusionar
        result = merge_enaho_modules(modulos=dataframes, nivel=args.level)

        # Guardar resultado
        if args.output.endswith(".csv"):
            result.to_csv(args.output, index=False)
        elif args.output.endswith(".dta"):
            result.to_stata(args.output)
        else:
            result.to_parquet(args.output)

        print(f"Fusión completada: {len(result)} registros guardados en {args.output}")
        return 0
    except Exception as e:
        print(f"Error en fusión: {e}", file=sys.stderr)
        return 1


def analyze_command(args):
    """Ejecutar comando de análisis"""
    try:
        import pandas as pd

        from .null_analysis import analyze_null_patterns

        print(f"Analizando {args.file}...")

        # Leer archivo
        if args.file.endswith(".dta"):
            df = pd.read_stata(args.file)
        elif args.file.endswith(".csv"):
            df = pd.read_csv(args.file)
        else:
            df = pd.read_parquet(args.file)

        # Realizar análisis
        if args.report == "nulls":
            result = analyze_null_patterns(df)
            print(f"Análisis de valores nulos completado")
            print(f"Completitud general: {result['completeness']:.1%}")

        # Guardar reporte si se especifica
        if args.output:
            # Implementar guardado de reporte
            pass

        return 0
    except Exception as e:
        print(f"Error en análisis: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
