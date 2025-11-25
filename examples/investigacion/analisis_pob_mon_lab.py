    """
Análisis de Pobreza Monetaria y Laboral - ENAHO
================================================

Este script realiza un análisis completo de pobreza monetaria y laboral
utilizando datos de la ENAHO. Incluye:

1. Procesamiento de características individuales y de hogar
2. Cálculo de informalidad laboral
3. Agregación de datos a nivel de hogar
4. Análisis descriptivo ponderado
5. Selección de features para modelado
6. Análisis de multicolinealidad (VIF)

Ejemplo de Uso con Tablas CIIU y CNO
-------------------------------------

    # Importar funciones
    from analisis_pob_mon_lab import pipeline_completo, analisis_descriptivo_ponderado

    # Cargar datos con enahopy
    from enahopy.loader import ENAHODataDownloader
    from enahopy.merger import ENAHOModuleMerger

    downloader = ENAHODataDownloader(year=2024, quarter=1)
    downloader.download()

    # Cargar módulos de personas y hogares
    df_personas = merger.merge_modules(modules=[1, 2, 5], year=2024, quarter=1)
    df_hogares = pd.read_stata(downloader.get_file_path(34, 2024, 1))

    # Ejecutar pipeline CON tablas CIIU y CNO
    df_final = pipeline_completo(
        df_individuos=df_personas,
        df_hogares_enaho=df_hogares,
        ruta_ciiu='data/modulo_05_2024/enaho_tabla_ciiu_rev4.dta',  # ← Tabla CIIU
        ruta_cno='data/modulo_05_2024/enaho_tabla_cno_2015.dta'     # ← Tabla CNO
    )

    # Análisis descriptivo
    resultados = analisis_descriptivo_ponderado(df_final, 'es_pobre_monetario', peso='factor07')

Notas sobre Tablas CIIU y CNO
------------------------------
- Las tablas CIIU y CNO son OPCIONALES
- Se descargan automáticamente con los módulos 02 y 05 de ENAHO
- CIIU: Clasifica sectores económicos (agricultura, industria, etc.)
- CNO: Clasifica ocupaciones laborales (gerentes, técnicos, etc.)
- Si no se proporcionan, el análisis se ejecuta sin estas clasificaciones adicionales

Para más ejemplos detallados, ver: ejemplo_uso_completo.py

Autor: enahopy team
Fecha: 2024
"""

# ================================================================================
# IMPORTACIONES
# ================================================================================
import warnings
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.stats import chi2_contingency, pointbiserialr
from statsmodels.stats.outliers_influence import variance_inflation_factor

warnings.filterwarnings('ignore')

# ================================================================================
# CONSTANTES GLOBALES
# ================================================================================
HOGAR_KEYS = ['conglome', 'vivienda', 'hogar']

# Mapeo educación a años
EDUCACION_ANIOS_MAP = {
    'sin nivel': 0,
    'educacion inicial': 0,
    'primaria incompleta': 0,
    'primaria completa': 6,
    'secundaria incompleta': 6,
    'secundaria completa': 11,
    'superior no universitaria incompleta': 11,
    'superior no universitaria completa': 14,
    'superior universitaria incompleta': 11,
    'superior universitaria completa': 16,
    'maestria/doctorado': 18,
    'basica especial': 6
}

# Grupos de ocupación para clasificación
OCUPACION_GRUPOS = {
    "Directivos y gerentes": ["director", "gerente", "ejecutivo", "presidente"],
    "Profesionales científicos e intelectuales": [
        "cientifico", "ingeniero", "medico", "profesor", "abogado", "economista",
        "arquitecto", "contador", "investigador"
    ],
    "Técnicos y profesionales de nivel medio": [
        "tecnico", "asistente", "agente", "inspector", "supervisor"
    ],
    "Personal de apoyo administrativo": [
        "secretaria", "oficinista", "auxiliar administrativo", "recepcionista"
    ],
    "Trabajadores de servicios y vendedores": [
        "vendedor", "comerciante", "cajero", "mesero", "cocinero", "guardia",
        "vigilante", "limpieza"
    ],
    "Agricultores y trabajadores agropecuarios": [
        "agricultor", "ganadero", "pescador", "forestal"
    ],
    "Operarios y artesanos": [
        "operario", "mecanico", "electricista", "carpintero", "albanil",
        "soldador", "costurera"
    ],
    "Operadores de máquinas y montadores": [
        "conductor", "chofer", "maquinista", "operador de maquinaria"
    ],
    "Ocupaciones elementales": [
        "peon", "ayudante", "cargador", "empacador", "recolector", "conserje"
    ]
}

# Variables globales para tablas de mapeo
CIIU_SECTOR_PRIMARIO = None
CNO_OCUPACIONES_ELEMENTALES = None

# ================================================================================
# FUNCIONES AUXILIARES
# ================================================================================

def normalizar_texto(serie: pd.Series) -> pd.Series:
    """
    Normaliza texto y remueve prefijos numéricos como '1.', '2.', etc.

    Parameters
    ----------
    serie : pd.Series
        Serie de texto a normalizar

    Returns
    -------
    pd.Series
        Serie normalizada
    """
    return serie.astype(str).str.lower().str.strip().str.replace(r'^\d+\.?\s*', '', regex=True)


def safe_numeric(serie: pd.Series) -> pd.Series:
    """
    Convierte serie a numérico de forma segura.

    Parameters
    ----------
    serie : pd.Series
        Serie a convertir

    Returns
    -------
    pd.Series
        Serie numérica con NaN donde no se pudo convertir
    """
    return pd.to_numeric(serie, errors='coerce')


def extraer_codigo_de_texto(serie: pd.Series, n_digitos: int = 4) -> pd.Series:
    """
    Extrae el código numérico del inicio de una serie de texto.
    Ejemplo: "5111. vendedores ambulantes" -> "5111"

    Parameters
    ----------
    serie : pd.Series
        Serie con códigos en texto
    n_digitos : int
        Número de dígitos a extraer

    Returns
    -------
    pd.Series
        Códigos extraídos
    """
    return serie.astype(str).str.extract(r'^(\d{' + str(n_digitos) + r'})', expand=False)


# ================================================================================
# FUNCIONES DE CARGA DE TABLAS DE MAPEO
# ================================================================================

def cargar_tabla_ciiu_oficial(ruta_archivo: str) -> List[str]:
    """
    Carga tabla CIIU oficial y extrae códigos del sector primario.

    Parameters
    ----------
    ruta_archivo : str
        Ruta al archivo .dta con códigos CIIU

    Returns
    -------
    List[str]
        Lista de códigos del sector primario
    """
    try:
        df_ciiu = pd.read_stata(ruta_archivo, convert_categoricals=False)
        df_ciiu['primer_digito'] = df_ciiu['codrev4'].str[:2]
        sectores_primarios = ['01', '02', '03', '05', '06', '07', '08', '09']
        codigos = df_ciiu[df_ciiu['primer_digito'].isin(sectores_primarios)]['codrev4'].tolist()
        print(f"  ✓ CIIU: {len(codigos)} códigos cargados")
        return codigos
    except Exception as e:
        print(f"  ✗ Error cargando CIIU: {e}")
        return []


def cargar_tabla_cno_oficial(ruta_archivo: str) -> List[int]:
    """
    Carga tabla CNO oficial y extrae códigos de ocupaciones elementales.

    Parameters
    ----------
    ruta_archivo : str
        Ruta al archivo .dta con códigos CNO

    Returns
    -------
    List[int]
        Lista de códigos de ocupaciones elementales (9000-9999)
    """
    try:
        df_cno = pd.read_stata(ruta_archivo, convert_categoricals=False)
        codigos = df_cno[(df_cno['código_cno_2015'] >= 9000) &
                        (df_cno['código_cno_2015'] < 10000)]['código_cno_2015'].tolist()
        print(f"  ✓ CNO: {len(codigos)} códigos cargados")
        return codigos
    except Exception as e:
        print(f"  ✗ Error cargando CNO: {e}")
        return []


def inicializar_tablas_mapeo(ruta_ciiu: str, ruta_cno: str) -> None:
    """
    Inicializa las tablas de mapeo globales CIIU y CNO.

    Parameters
    ----------
    ruta_ciiu : str
        Ruta al archivo CIIU
    ruta_cno : str
        Ruta al archivo CNO
    """
    global CIIU_SECTOR_PRIMARIO, CNO_OCUPACIONES_ELEMENTALES
    print("\n→ Cargando tablas CIIU y CNO...")
    CIIU_SECTOR_PRIMARIO = cargar_tabla_ciiu_oficial(ruta_ciiu)
    CNO_OCUPACIONES_ELEMENTALES = cargar_tabla_cno_oficial(ruta_cno)


# ================================================================================
# FUNCIONES DE PROCESAMIENTO - NIVEL INDIVIDUAL
# ================================================================================

def limpiar_variables_individuales(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convierte variables clave de 'object' a 'numeric' para poder usarlas en cálculos.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame con datos individuales

    Returns
    -------
    pd.DataFrame
        DataFrame con variables limpias
    """
    print("\n→ Limpiando variables individuales...")
    df = df.copy()

    # Variables numéricas a convertir
    vars_numericas = ['p208a', 'p513t', 'i524e1', 'p301b', 'p301c']

    for var in vars_numericas:
        if var in df.columns:
            df[var] = safe_numeric(df[var])
            print(f"  ✓ {var}: {df[var].notna().sum()} valores válidos")

    return df


def calcular_informalidad_vectorizada(df: pd.DataFrame) -> pd.Series:
    """
    Clasifica a trabajadores como formales (0) o informales (1) de forma vectorizada.

    Criterios:
    - Independientes/Patronos: informales si no están registrados
    - Empleados/Obreros: informales si no están afiliados a seguro
    - Familiares/Hogar: siempre informales

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame con datos individuales

    Returns
    -------
    pd.Series
        Serie con clasificación de informalidad (0=formal, 1=informal, NaN=no ocupado)
    """
    print("  • Calculando informalidad laboral...")

    # Normalizar variables
    ocu_norm = normalizar_texto(df['ocu500']) if 'ocu500' in df.columns else pd.Series('', index=df.index)
    p507_norm = normalizar_texto(df['p507']) if 'p507' in df.columns else pd.Series('', index=df.index)
    p510a1_norm = normalizar_texto(df['p510a1']) if 'p510a1' in df.columns else pd.Series('', index=df.index)
    p558a5_norm = normalizar_texto(df['p558a5']) if 'p558a5' in df.columns else pd.Series('', index=df.index)

    # Inicializar serie de informalidad
    informalidad = pd.Series(np.nan, index=df.index)
    es_ocupado = ocu_norm.str.contains('ocupado', na=False)

    if es_ocupado.sum() == 0:
        print("    ⚠ No hay ocupados en el dataset")
        return informalidad

    # INDEPENDIENTES Y PATRONOS
    es_independiente = (p507_norm.str.contains('independiente', na=False) |
                       p507_norm.str.contains('patrono', na=False) |
                       p507_norm.str.contains('empleador', na=False))
    no_registrado = p510a1_norm.str.contains('no esta registrado', na=False)
    informalidad.loc[es_ocupado & es_independiente] = no_registrado[es_ocupado & es_independiente].astype(int)

    # EMPLEADOS Y OBREROS
    es_empleado = (p507_norm.str.contains('empleado', na=False) |
                   p507_norm.str.contains('obrero', na=False))
    no_afiliado = (p558a5_norm.str.contains('no esta afiliado', na=False) |
                   p558a5_norm.str.contains('no esta afiiado', na=False))
    informalidad.loc[es_ocupado & es_empleado] = no_afiliado[es_ocupado & es_empleado].astype(int)

    # FAMILIARES Y TRABAJADORES DEL HOGAR
    es_familiar = (p507_norm.str.contains('familiar', na=False) |
                   p507_norm.str.contains('hogar', na=False))
    informalidad.loc[es_ocupado & es_familiar] = 1

    # Estadísticas
    n_informales = (informalidad == 1).sum()
    n_ocupados = es_ocupado.sum()
    pct = (n_informales/n_ocupados*100) if n_ocupados > 0 else 0
    print(f"    ✓ {n_informales}/{n_ocupados} informales ({pct:.1f}%)")

    return informalidad


def calcular_anios_escolaridad(df: pd.DataFrame) -> pd.Series:
    """
    Calcula años de escolaridad según nivel educativo reportado.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame con datos individuales

    Returns
    -------
    pd.Series
        Años de escolaridad calculados
    """
    print("  • Calculando años de escolaridad...")

    if 'p301a' not in df.columns:
        print("    ⚠ Variable p301a no encontrada")
        return pd.Series(0, index=df.index)

    p301a_norm = normalizar_texto(df['p301a'])
    anio = safe_numeric(df['p301b']) if 'p301b' in df.columns else pd.Series(0, index=df.index)
    grado = safe_numeric(df['p301c']) if 'p301c' in df.columns else pd.Series(0, index=df.index)

    anios_esc = pd.Series(0, index=df.index, dtype=float)

    # Mapear según texto
    for nivel_texto, anios_base in EDUCACION_ANIOS_MAP.items():
        mask = p301a_norm.str.contains(nivel_texto.replace(' ', '.*'), na=False, regex=True)
        anios_esc.loc[mask] = anios_base

        # Agregar años adicionales para incompletos
        if 'incompleta' in nivel_texto:
            anios_adicionales = anio.fillna(0) + grado.fillna(0)
            if 'primaria' in nivel_texto:
                anios_adicionales = anios_adicionales.clip(upper=5)
            elif 'secundaria' in nivel_texto:
                anios_adicionales = anios_adicionales.clip(upper=4)
            elif 'superior' in nivel_texto:
                anios_adicionales = anios_adicionales.clip(upper=4)
            anios_esc.loc[mask] += anios_adicionales.loc[mask]

    anios_esc = anios_esc.clip(0, 20)
    print(f"    ✓ Media: {anios_esc.mean():.1f} años")

    return anios_esc


def crear_caracteristicas_individuales(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea características derivadas a nivel individual para posterior agregación.

    Características creadas:
    - ocupinf: informalidad laboral
    - tiene_discapacidad: al menos una limitación
    - tiene_seguro: al menos un seguro
    - es_nino, es_adulto_mayor, es_edad_trabajar: grupos etarios
    - es_ocupado: indicador de ocupación
    - anios_escolaridad: años de educación

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame con datos individuales ya limpiados

    Returns
    -------
    pd.DataFrame
        DataFrame con características adicionales
    """
    print("\n→ Creando características individuales...")
    df = df.copy()

    # 1. Informalidad (ocupados solamente)
    df['ocupinf'] = calcular_informalidad_vectorizada(df)

    # 2. Discapacidad: 1 si tiene AL MENOS UNA limitación
    print("  • Identificando personas con discapacidad...")

    # Versión más compacta (mismo resultado):
    disc_cols = ['p401h1', 'p401h2', 'p401h3', 'p401h4', 'p401h5', 'p401h6']

    # Crear condición combinada directamente
    condicion_discapacidad = False
    for col in disc_cols:
        if col in df.columns:
            condicion = df[col].astype(str).str.lower().str.strip() == 'si'
            condicion_discapacidad = condicion_discapacidad | condicion

    df['tiene_discapacidad'] = condicion_discapacidad.astype(int)

    print(
        f"    ✓ {df['tiene_discapacidad'].sum():,} personas con discapacidad ({df['tiene_discapacidad'].mean() * 100:.1f}%)")


    # 3. Seguro: 1 si tiene AL MENOS UN seguro
    print("  • Identificando personas con seguro de salud...")

    seguro_cols = ['p4191', 'p4192', 'p4193', 'p4194', 'p4195', 'p4196', 'p4197', 'p4198']
    seguro_nombres = {
        'p4191': 'essalud',
        'p4192': 'seguro_privado',
        'p4193': 'entidad_prestadora',
        'p4194': 'seguro_ffaa_policial',
        'p4195': 'sis',
        'p4196': 'seguro_universitario',
        'p4197': 'seguro_escolar_privado',
        'p4198': 'otro_seguro'
    }

    # Verificar si existen columnas de seguro
    cols_seguro_disponibles = [col for col in seguro_cols if col in df.columns]

    if not cols_seguro_disponibles:
        print("    ⚠ ADVERTENCIA: No hay columnas de seguro en los datos")
        print("      Las variables p4191-p4198 vienen del MÓDULO 04 (Salud) de ENAHO")
        print("      Incluye el módulo 04 al cargar los datos:")
        print("      merger.merge_modules(modules=[1, 2, 4, 5], ...)")
        df['tiene_seguro'] = 0
        print(f"    ✓ Asignando tiene_seguro = 0 para todos")
    else:
        print(f"    ✓ Encontradas {len(cols_seguro_disponibles)} columnas de seguro: {', '.join(cols_seguro_disponibles)}")

        # DEBUG: Mostrar valores únicos de la primera columna
        col_ejemplo = cols_seguro_disponibles[0]
        valores_unicos = df[col_ejemplo].value_counts(dropna=False).head(5)
        print(f"    DEBUG - Valores únicos en {col_ejemplo}:")
        for val, count in valores_unicos.items():
            print(f"      {repr(val):>15} : {count:>8,} ({count/len(df)*100:>5.1f}%)")

        # Crear variables individuales para cada tipo de seguro
        for col, nombre in seguro_nombres.items():
            if col in df.columns:
                # Los datos de ENAHO usan estos formatos:
                # - 'no' = no tiene seguro
                # - nombre del seguro (ej: 'essalud', 'sis') = SI tiene
                # - nan = no respondió

                # Convertir a string y limpiar
                val_str = df[col].astype(str).str.strip().str.lower()

                # OPCION A: Tiene seguro si el valor NO es 'no', 'nan' o vacio
                condicion = (
                    (val_str != 'no') &
                    (val_str != 'nan') &
                    (val_str != '') &
                    (df[col].notna())
                )

                # OPCION B (alternativa): Si es numerico 1 o string '1' o 'si'
                # (por compatibilidad con otros formatos de ENAHO)
                try:
                    valores_numericos = pd.to_numeric(df[col], errors='coerce')
                    condicion = condicion | (valores_numericos == 1)
                except:
                    pass

                df[f'tiene_{nombre}'] = condicion.astype(int)

                # Debug: mostrar cuántos tienen este seguro
                n_con_seguro = df[f'tiene_{nombre}'].sum()
                print(f"      • {nombre}: {n_con_seguro:,} personas ({n_con_seguro/len(df)*100:.1f}%)")

        # Variable general: tiene AL MENOS UN seguro
        cols_seguro = [f'tiene_{nombre}' for col, nombre in seguro_nombres.items() if col in df.columns]
        df['tiene_seguro'] = (df[cols_seguro].sum(axis=1) > 0).astype(int)

        print(f"    ✓ TOTAL con al menos un seguro: {df['tiene_seguro'].sum():,} personas ({df['tiene_seguro'].mean() * 100:.1f}%)")


    # 4. Grupos de Edad (para ratio de dependencia)
    print("  • Clasificando grupos etarios...")
    df['es_nino'] = np.where(df['p208a'] < 14, 1, 0)
    df['es_adulto_mayor'] = np.where(df['p208a'] >= 65, 1, 0)
    df['es_edad_trabajar'] = np.where((df['p208a'] >= 14) & (df['p208a'] < 65), 1, 0)
    print(f"    ✓ Niños: {df['es_nino'].sum()}, Edad trabajar: {df['es_edad_trabajar'].sum()}, Adultos mayores: {df['es_adulto_mayor'].sum()}")

    # 5. Es ocupado (binaria para facilitar agregación)
    df['es_ocupado'] = np.where(
        df['ocu500'].astype(str).str.lower().str.strip() == 'ocupado',
        1,
        0
    )

    # 6. Años de escolaridad
    df['anios_escolaridad'] = calcular_anios_escolaridad(df)

    # Limpiar columnas auxiliares
    cols_drop = ['p401h1_str', 'p401h6_str', 'p4191_str', 'p4198_str']
    df.drop(columns=[col for col in cols_drop if col in df.columns], inplace=True)

    return df


# ================================================================================
# FUNCIONES DE PROCESAMIENTO - NIVEL HOGAR
# ================================================================================

def extraer_jefe_hogar_completo(df_individuos: pd.DataFrame) -> pd.DataFrame:
    """
    Filtra el DataFrame de individuos para obtener solo al jefe de hogar
    y prepara sus características para un merge a nivel de hogar.

    Parameters
    ----------
    df_individuos : pd.DataFrame
        DataFrame con datos individuales

    Returns
    -------
    pd.DataFrame
        DataFrame con características del jefe de hogar
    """
    print("\n→ Extrayendo características del Jefe de Hogar...")

    # Filtrar jefes (p203 == 'jefe/jefa')
    df_jefe = df_individuos[
        df_individuos['p203'].astype(str).str.lower().str.strip() == 'jefe/jefa'
    ].copy()

    print(f"  ✓ {len(df_jefe)} jefes de hogar identificados")

    # Seleccionar y renombrar variables básicas (incluir ocupación y sector si existen)
    columnas_jefe = HOGAR_KEYS + ['p208a', 'p207', 'p301a', 'anios_escolaridad', 'p505r4', 'p506r4', 'p511a']

    # Verificar que existan las columnas
    columnas_disponibles = [col for col in columnas_jefe if col in df_jefe.columns]
    df_jefe = df_jefe[columnas_disponibles].copy()

    # Renombrar
    rename_dict = {
        'p208a': 'jefe_edad',
        'p207': 'jefe_sexo',
        'p301a': 'jefe_educacion_original',
        'anios_escolaridad': 'jefe_anios_escolaridad',
        'p505r4': 'jefe_ocupacion_cno',
        'p506r4': 'jefe_sector_ciiu',
        'p511a': 'jefe_tipo_contrato'
    }
    df_jefe = df_jefe.rename(columns={k: v for k, v in rename_dict.items() if k in df_jefe.columns})

    # Agrupar educación del jefe en 3 categorías
    def agrupar_educacion(x):
        x_str = str(x).lower().strip()
        if any(word in x_str for word in ['superior', 'universitaria', 'maestria', 'doctorado']):
            return 'Alta'
        elif 'secundaria' in x_str:
            return 'Media'
        elif any(word in x_str for word in ['primaria', 'inicial', 'sin nivel', 'basica']):
            return 'Baja'
        else:
            return 'Baja'  # Por defecto asignamos Baja a missings

    if 'jefe_educacion_original' in df_jefe.columns:
        df_jefe['jefe_educacion'] = df_jefe['jefe_educacion_original'].apply(agrupar_educacion)
        print(f"  Distribución educación jefes:")
        print(df_jefe['jefe_educacion'].value_counts().to_dict())

    # Crear indicador sexo mujer
    if 'jefe_sexo' in df_jefe.columns:
        df_jefe['jefe_mujer'] = np.where(
            df_jefe['jefe_sexo'].astype(str).str.lower().str.contains('mujer', na=False),
            1,
            0
        )

    # Clasificar ocupación del jefe usando la tabla CNO o la función de clasificación
    if 'jefe_ocupacion_cno' in df_jefe.columns:
        print("  • Clasificando ocupación del jefe de hogar...")
        df_jefe['jefe_grupo_ocupacional'] = df_jefe['jefe_ocupacion_cno'].apply(clasificar_ocupacion)
        print(f"    ✓ Ocupaciones clasificadas")

        # Mostrar distribución
        dist_ocup = df_jefe['jefe_grupo_ocupacional'].value_counts()
        if len(dist_ocup) <= 5:
            print(f"    Distribución: {dist_ocup.to_dict()}")

    # Clasificar sector económico del jefe usando la tabla CIIU o la función de clasificación
    if 'jefe_sector_ciiu' in df_jefe.columns:
        print("  • Clasificando sector económico del jefe de hogar...")
        df_jefe['jefe_grupo_sectorial'] = df_jefe['jefe_sector_ciiu'].apply(clasificar_ciuu)
        print(f"    ✓ Sectores clasificados")

        # Mostrar distribución
        dist_sector = df_jefe['jefe_grupo_sectorial'].value_counts()
        if len(dist_sector) <= 5:
            print(f"    Distribución: {dist_sector.to_dict()}")

    return df_jefe


def agregar_a_nivel_hogar(df_individuos: pd.DataFrame) -> pd.DataFrame:
    """
    Toma el DataFrame de individuos ya procesado y lo resume a nivel de hogar.
    Incluye características calculadas Y modas de variables originales.

    Parameters
    ----------
    df_individuos : pd.DataFrame
        DataFrame con datos individuales procesados

    Returns
    -------
    pd.DataFrame
        DataFrame agregado a nivel de hogar
    """
    print("\n→ Agregando características a nivel de hogar...")

    df = df_individuos.copy()

    # Convertir llaves a string para evitar problemas en groupby
    for col in HOGAR_KEYS:
        if col in df.columns:
            df[col] = df[col].astype(str)

    # Definir agregaciones
    agregaciones = {
        # CARACTERÍSTICAS CALCULADAS (sumas y conteos)
        'codperso': 'count',  # n_personas (miembros del hogar)
        'es_nino': 'sum',  # n_ninos
        'es_adulto_mayor': 'sum',  # n_adultos_mayores
        'es_edad_trabajar': 'sum',  # n_edad_trabajar
        'es_ocupado': 'sum',  # n_ocupados
        'ocupinf': 'sum',  # n_informales (solo cuenta los 1s, NaN son ignorados)
        'tiene_discapacidad': 'sum',  # n_discapacitados
        'tiene_seguro': 'sum',  # n_asegurados

        # VARIABLES NUMÉRICAS CONTINUAS (promedios o sumas)
        'p513t': 'mean',  # horas_trabajo_promedio
        'i524e1': 'sum',  # ingreso_laboral_total
        'p208a': 'mean',  # edad_promedio_hogar
    }

    # Agregar variables opcionales si existen
    vars_opcionales = {
        'anios_escolaridad': 'mean',
        'p203b': lambda x: x.mode()[0] if not x.mode().empty else np.nan,
        'p207': lambda x: x.mode()[0] if not x.mode().empty else np.nan,
        'p507': lambda x: x.mode()[0] if not x.mode().empty else np.nan,
        'p510a1': lambda x: x.mode()[0] if not x.mode().empty else np.nan,
        'p512a': lambda x: x.mode()[0] if not x.mode().empty else np.nan,
        'p558a5': lambda x: x.mode()[0] if not x.mode().empty else np.nan,
        'p301a': lambda x: x.mode()[0] if not x.mode().empty else np.nan,
        'p4191': lambda x: x.mode()[0] if not x.mode().empty else np.nan,
        'p4198': lambda x: x.mode()[0] if not x.mode().empty else np.nan,
        'p401h1': lambda x: x.mode()[0] if not x.mode().empty else np.nan,
        'p401h6': lambda x: x.mode()[0] if not x.mode().empty else np.nan,
        'ocu500': lambda x: x.mode()[0] if not x.mode().empty else np.nan,
    }

    for var, agg_func in vars_opcionales.items():
        if var in df.columns:
            agregaciones[var] = agg_func

    # Ejecutar agregación
    df_agregado = df.groupby(HOGAR_KEYS).agg(agregaciones).reset_index()

    # Renombrar columnas para claridad
    rename_dict = {
        'codperso': 'n_personas',
        'es_nino': 'n_ninos',
        'es_adulto_mayor': 'n_adultos_mayores',
        'es_edad_trabajar': 'n_edad_trabajar',
        'es_ocupado': 'n_ocupados',
        'ocupinf': 'n_informales',
        'tiene_discapacidad': 'n_discapacitados',
        'tiene_seguro': 'n_asegurados',
        'p513t': 'horas_trabajo_promedio',
        'i524e1': 'ingreso_laboral_total',
        'p208a': 'edad_promedio_hogar',
        'anios_escolaridad': 'anios_escolaridad_promedio',
    }

    # Agregar sufijo _moda a variables categóricas
    for var in ['p203b', 'p207', 'p507', 'p510a1', 'p512a', 'p558a5',
                'p301a', 'p4191', 'p4198', 'p401h1', 'p401h6', 'ocu500']:
        if var in df_agregado.columns:
            rename_dict[var] = f'{var}_moda'

    df_agregado = df_agregado.rename(columns=rename_dict)

    print(f"  ✓ Hogares únicos: {len(df_agregado)}")
    print(f"  ✓ Columnas en dataset agregado: {len(df_agregado.columns)}")

    return df_agregado


def agregar_caracteristicas_vivienda(df_hogar: pd.DataFrame,
                                     df_hogares_enaho: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega características de vivienda desde el módulo de hogares.

    Parameters
    ----------
    df_hogar : pd.DataFrame
        DataFrame agregado a nivel de hogar
    df_hogares_enaho : pd.DataFrame
        DataFrame del módulo de hogares de ENAHO

    Returns
    -------
    pd.DataFrame
        DataFrame con características de vivienda agregadas
    """
    print("\n→ Agregando características de vivienda...")

    # Asegurar que las llaves sean string
    for col in HOGAR_KEYS:
        if col in df_hogar.columns:
            df_hogar[col] = df_hogar[col].astype(str)
        if col in df_hogares_enaho.columns:
            df_hogares_enaho[col] = df_hogares_enaho[col].astype(str)

    # Merge
    df_merged = df_hogar.merge(
        df_hogares_enaho,
        on=HOGAR_KEYS,
        how='left',
        suffixes=('', '_hogar')
    )

    print(f"  ✓ Merge completado: {len(df_merged)} registros")

    return df_merged


def calcular_ratios_hogar(df_hogar: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula ratios y tasas derivadas de las características agregadas.

    Parameters
    ----------
    df_hogar : pd.DataFrame
        DataFrame a nivel de hogar

    Returns
    -------
    pd.DataFrame
        DataFrame con ratios calculados
    """
    print("\n→ Calculando ratios y tasas a nivel hogar...")
    df = df_hogar.copy()

    # Ratio de dependencia
    df['ratio_dependencia'] = np.where(
        df['n_edad_trabajar'] > 0,
        (df['n_ninos'] + df['n_adultos_mayores']) / df['n_edad_trabajar'],
        np.nan
    )

    # Tasa de ocupación del hogar
    if 'mieperho' in df.columns:
        df['tasa_ocupacion_hogar'] = df['n_ocupados'] / df['mieperho']
    else:
        df['tasa_ocupacion_hogar'] = df['n_ocupados'] / df['n_personas']

    # Tasa de informalidad del hogar
    df['tasa_informalidad_hogar'] = np.where(
        df['n_ocupados'] > 0,
        df['n_informales'] / df['n_ocupados'],
        np.nan
    )

    # Proporción de asegurados
    df['prop_asegurados'] = df['n_asegurados'] / df['n_personas']

    # Proporción de discapacitados
    df['prop_discapacitados'] = df['n_discapacitados'] / df['n_personas']

    # Ingreso laboral per cápita (si existe mieperho)
    if 'mieperho' in df.columns:
        df['ingreso_laboral_percapita'] = df['ingreso_laboral_total'] / df['mieperho']

    # Ingreso por hora trabajada
    if 'horas_trabajo_promedio' in df.columns:
        df['ingreso_por_hora'] = np.where(
            df['horas_trabajo_promedio'] > 0,
            df['ingreso_laboral_total'] / (df['horas_trabajo_promedio'] * df['n_ocupados'] * 4),
            np.nan
        )

    # Cargas demográficas per cápita
    if 'mieperho' in df.columns:
        df['carga_ninos'] = df['n_ninos'] / df['mieperho']
        df['carga_adultos_mayores'] = df['n_adultos_mayores'] / df['mieperho']
        df['ocupados_per_capita'] = df['n_ocupados'] / df['mieperho']

    print(f"  ✓ Ratio dependencia promedio: {df['ratio_dependencia'].mean():.2f}")
    print(f"  ✓ Tasa informalidad promedio: {df['tasa_informalidad_hogar'].mean()*100:.1f}%")
    print(f"  ✓ Proporción asegurados promedio: {df['prop_asegurados'].mean()*100:.1f}%")

    return df


def calcular_variables_objetivo(df_merged: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula Y1 (pobreza monetaria) y Y2 (pobreza laboral) después del merge.
    REQUIERE: columnas 'pobreza', 'linea', 'mieperho' del módulo hogares.

    Parameters
    ----------
    df_merged : pd.DataFrame
        DataFrame con merge de características de hogar y módulo hogares

    Returns
    -------
    pd.DataFrame
        DataFrame con variables objetivo calculadas
    """
    print("\n→ Calculando variables objetivo (Y1, Y2)...")
    df = df_merged.copy()

    # Y1: Es pobre monetario (pobre extremo o pobre no extremo)
    if 'pobreza' in df.columns:
        df['pobreza_str'] = df['pobreza'].astype(str).str.lower().str.strip()
        df['es_pobre_monetario'] = np.where(
            df['pobreza_str'].isin(['pobre extremo', 'pobre no extremo']),
            1,
            0
        )
        df.drop(columns=['pobreza_str'], inplace=True)
        print(f"  ✓ Pobreza monetaria: {df['es_pobre_monetario'].sum()} hogares ({df['es_pobre_monetario'].mean()*100:.1f}%)")
    else:
        print("  ⚠ Variable 'pobreza' no encontrada")

    # Y2: Pobreza laboral (ingreso laboral per cápita < línea de pobreza)
    if all(col in df.columns for col in ['ingreso_laboral_total', 'mieperho', 'linea']):
        df['ingreso_laboral_percapita'] = df['ingreso_laboral_total'] / df['mieperho']
        df['pobreza_laboral'] = np.where(
            df['ingreso_laboral_percapita'] < df['linea'],
            1,
            0
        )
        print(f"  ✓ Pobreza laboral: {df['pobreza_laboral'].sum()} hogares ({df['pobreza_laboral'].mean()*100:.1f}%)")
    else:
        print("  ⚠ Variables para pobreza laboral no encontradas")

    return df


# ================================================================================
# FUNCIÓN DE PIPELINE COMPLETO
# ================================================================================

def pipeline_completo(df_individuos: pd.DataFrame,
                     df_hogares_enaho: pd.DataFrame,
                     ruta_ciiu: str = None,
                     ruta_cno: str = None) -> pd.DataFrame:
    """
    Pipeline completo de procesamiento de datos ENAHO.

    Este pipeline ejecuta en secuencia:
    1. Limpieza de variables individuales
    2. Creación de características individuales
    3. Extracción de características del jefe de hogar
    4. Agregación a nivel de hogar
    5. Merge con módulo de hogares
    6. Cálculo de ratios y tasas
    7. Cálculo de variables objetivo

    Parameters
    ----------
    df_individuos : pd.DataFrame
        DataFrame con datos individuales (módulo persona)
    df_hogares_enaho : pd.DataFrame
        DataFrame con datos de hogares (módulo sumaria/hogares)
    ruta_ciiu : str, optional
        Ruta a tabla CIIU oficial
    ruta_cno : str, optional
        Ruta a tabla CNO oficial

    Returns
    -------
    pd.DataFrame
        DataFrame final procesado y listo para análisis/modelado
    """
    print("\n" + "="*80)
    print("INICIANDO PIPELINE COMPLETO DE PROCESAMIENTO")
    print("="*80)

    # Opcional: cargar tablas de mapeo
    if ruta_ciiu and ruta_cno:
        inicializar_tablas_mapeo(ruta_ciiu, ruta_cno)

    # 1. Limpieza
    df_proc = limpiar_variables_individuales(df_individuos)

    # 2. Características individuales
    df_proc = crear_caracteristicas_individuales(df_proc)

    # 3. Extraer jefe de hogar
    df_jefe = extraer_jefe_hogar_completo(df_proc)

    # 4. Agregación a nivel hogar
    df_hogar = agregar_a_nivel_hogar(df_proc)

    # 5. Merge con jefe de hogar
    df_hogar = df_hogar.merge(df_jefe, on=HOGAR_KEYS, how='left')

    # 6. Merge con módulo hogares
    df_hogar = agregar_caracteristicas_vivienda(df_hogar, df_hogares_enaho)

    # 7. Calcular ratios
    df_hogar = calcular_ratios_hogar(df_hogar)

    # 8. Calcular variables objetivo
    df_hogar = calcular_variables_objetivo(df_hogar)

    print("\n" + "="*80)
    print("PIPELINE COMPLETADO")
    print(f"Dataset final: {df_hogar.shape[0]} hogares × {df_hogar.shape[1]} variables")
    print("="*80)

    return df_hogar


# ================================================================================
# FUNCIONES DE ANÁLISIS DESCRIPTIVO
# ================================================================================

def weighted_avg(df: pd.DataFrame, values_col: str, weights_col: str = 'factor07') -> float:
    """
    Calcula la media ponderada de una columna.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame con los datos
    values_col : str
        Nombre de la columna con los valores a promediar
    weights_col : str
        Nombre de la columna con los pesos (factor de expansión)

    Returns
    -------
    float
        Media ponderada
    """
    d = df[values_col]
    w = df[weights_col]

    valid_mask = d.notna() & w.notna()
    d_valid = d[valid_mask]
    w_valid = w[valid_mask]

    if len(d_valid) == 0 or w_valid.sum() == 0:
        return np.nan

    return (d_valid * w_valid).sum() / w_valid.sum()


def weighted_std(df: pd.DataFrame, values_col: str, weights_col: str = 'factor07') -> float:
    """
    Calcula la desviación estándar ponderada.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame con los datos
    values_col : str
        Nombre de la columna con los valores
    weights_col : str
        Nombre de la columna con los pesos

    Returns
    -------
    float
        Desviación estándar ponderada
    """
    d = df[values_col]
    w = df[weights_col]

    valid_mask = d.notna() & w.notna()
    d_valid = d[valid_mask]
    w_valid = w[valid_mask]

    if len(d_valid) == 0 or w_valid.sum() == 0:
        return np.nan

    mean_w = weighted_avg(df, values_col, weights_col)
    variance = (w_valid * (d_valid - mean_w)**2).sum() / w_valid.sum()
    return np.sqrt(variance)


def weighted_proportion(df: pd.DataFrame, cat_col: str, weights_col: str = 'factor07') -> pd.Series:
    """
    Calcula proporciones ponderadas para variables categóricas.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame con los datos
    cat_col : str
        Columna categórica
    weights_col : str
        Columna con pesos

    Returns
    -------
    pd.Series
        Proporciones ponderadas por categoría
    """
    grouped = df.groupby(cat_col)[weights_col].sum()
    total_weight = df[weights_col].sum()

    if total_weight == 0:
        return pd.Series()

    return (grouped / total_weight * 100).sort_values(ascending=False)


def analisis_descriptivo_ponderado(df: pd.DataFrame,
                                   var_pobreza: str,
                                   peso: str = 'factor07') -> Dict:
    """
    Realiza análisis descriptivo comparativo ponderado según condición de pobreza.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset con todas las variables
    var_pobreza : str
        Variable de pobreza a analizar ('es_pobre_monetario' o 'pobreza_laboral')
    peso : str
        Variable de ponderación (factor de expansión)

    Returns
    -------
    dict
        Diccionario con DataFrames de resultados
    """
    print("="*80)
    print(f"ANÁLISIS DESCRIPTIVO PONDERADO: {var_pobreza.upper()}")
    print("="*80)

    # Dividir en grupos
    df_pobre = df[df[var_pobreza] == 1].copy()
    df_no_pobre = df[df[var_pobreza] == 0].copy()

    # Calcular población expandida
    n_total = df[peso].sum()
    n_pobre = df_pobre[peso].sum()
    n_no_pobre = df_no_pobre[peso].sum()

    print(f"\n POBLACIÓN EXPANDIDA:")
    print(f"  • Total hogares: {n_total:,.0f}")
    print(f"  • Hogares pobres: {n_pobre:,.0f} ({n_pobre/n_total*100:.1f}%)")
    print(f"  • Hogares no pobres: {n_no_pobre:,.0f} ({n_no_pobre/n_total*100:.1f}%)")

    resultados = {}

    # Variables numéricas a analizar
    vars_numericas = [
        'mieperho', 'n_ninos', 'n_adultos_mayores', 'n_ocupados',
        'tasa_ocupacion_hogar', 'tasa_informalidad_hogar',
        'prop_asegurados', 'ratio_dependencia', 'jefe_edad'
    ]

    resultados_num = []
    for var in vars_numericas:
        if var in df.columns:
            media_pobre = weighted_avg(df_pobre, var, peso)
            media_no_pobre = weighted_avg(df_no_pobre, var, peso)

            dif_abs = media_pobre - media_no_pobre
            dif_rel = (dif_abs / media_no_pobre * 100) if media_no_pobre != 0 else np.nan

            resultados_num.append({
                'Variable': var,
                'Media_Pobre': media_pobre,
                'Media_No_Pobre': media_no_pobre,
                'Diferencia': dif_abs,
                'Diferencia_%': dif_rel
            })

    df_numericas = pd.DataFrame(resultados_num)
    resultados['numericas'] = df_numericas

    print("\n TOP BRECHAS NUMÉRICAS:")
    print(df_numericas.nlargest(5, 'Diferencia_%')[['Variable', 'Diferencia_%']])

    return resultados


# ================================================================================
# FUNCIONES DE SELECCIÓN DE FEATURES
# ================================================================================

def calcular_vif(df: pd.DataFrame, variables: List[str]) -> pd.DataFrame:
    """
    Calcula VIF (Variance Inflation Factor) para detectar multicolinealidad.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame con datos
    variables : List[str]
        Lista de variables numéricas a analizar

    Returns
    -------
    pd.DataFrame
        DataFrame con resultados de VIF
    """
    print("\n→ Calculando VIF (Variance Inflation Factor)...")

    # Preparar datos
    df_vif = df[variables].copy()
    df_vif = df_vif.dropna()

    if len(df_vif) == 0:
        print("  ⚠ No hay datos válidos para calcular VIF")
        return pd.DataFrame()

    # Calcular VIF
    vif_data = []
    for i, col in enumerate(df_vif.columns):
        try:
            vif = variance_inflation_factor(df_vif.values, i)
            vif_data.append({'Variable': col, 'VIF': vif})
        except:
            vif_data.append({'Variable': col, 'VIF': np.nan})

    df_vif_results = pd.DataFrame(vif_data).sort_values('VIF', ascending=False)

    print(f"  ✓ VIF calculado para {len(df_vif_results)} variables")
    print("\n  Variables con VIF > 10 (problemáticas):")
    problematicas = df_vif_results[df_vif_results['VIF'] > 10]
    if len(problematicas) > 0:
        print(problematicas)
    else:
        print("  Ninguna variable con VIF > 10")

    return df_vif_results


# ================================================================================
# FUNCIONES DE CLASIFICACIÓN (AUXILIARES)
# ================================================================================

def clasificar_ocupacion(ocupacion: str) -> str:
    """
    Clasifica ocupación según grupos definidos.

    Parameters
    ----------
    ocupacion : str
        Descripción de ocupación

    Returns
    -------
    str
        Grupo ocupacional
    """
    if pd.isna(ocupacion):
        return "No especificado"

    ocupacion_lower = ocupacion.lower()
    for grupo, patrones in OCUPACION_GRUPOS.items():
        if any(pat in ocupacion_lower for pat in patrones):
            return grupo
    return "Otros / No clasificado"


def clasificar_ciuu(descripcion: str) -> str:
    """
    Clasifica sector económico según descripción CIIU.

    Parameters
    ----------
    descripcion : str
        Descripción del sector

    Returns
    -------
    str
        Grupo sectorial
    """
    if pd.isna(descripcion):
        return "No especificado"

    desc = descripcion.lower()

    if any(p in desc for p in ["fabricación", "elaboración", "procesamiento", "industria", "producción"]):
        return "Industria y manufactura"
    elif any(p in desc for p in ["cultivo", "cría", "pesca", "acuicultura", "silvicultura", "ganadería"]):
        return "Agricultura, ganadería y pesca"
    elif any(p in desc for p in ["extracción", "petróleo", "gas", "electricidad", "agua", "desechos", "minería"]):
        return "Minería, energía y servicios básicos"
    elif any(p in desc for p in ["construcción", "instalación", "fontanería", "mantenimiento", "edificios"]):
        return "Construcción e infraestructura"
    elif any(p in desc for p in ["transporte", "almacenamiento", "mensajería", "correo", "logística"]):
        return "Transporte y almacenamiento"
    elif any(p in desc for p in ["venta", "comercio", "reparación", "minorista", "mayorista", "retail"]):
        return "Comercio y reparación"
    elif any(p in desc for p in ["banco", "fondo", "seguro", "consultoría", "jurídico", "contable"]):
        return "Servicios empresariales, financieros y profesionales"
    elif any(p in desc for p in ["peluquería", "limpieza", "funeral", "asociación", "religioso"]):
        return "Servicios personales y comunitarios"
    elif any(p in desc for p in ["educación", "enseñanza", "hospital", "médico", "deporte", "arte"]):
        return "Educación, salud y cultura"
    elif any(p in desc for p in ["administración", "defensa", "seguridad", "gobierno", "público"]):
        return "Administración pública y defensa"
    else:
        return "Otros"


# ================================================================================
# FUNCIÓN PRINCIPAL
# ================================================================================

def main():
    """
    Función principal de demostración del pipeline.

    NOTA: Esta función requiere que existan los datos cargados.
    Modifica las rutas según tu configuración.
    """
    print("\n" + "="*80)
    print("ANÁLISIS DE POBREZA MONETARIA Y LABORAL - ENAHO")
    print("="*80)

    # Configuración de visualización
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    pd.set_option('display.float_format', lambda x: f'{x:.2f}')

    print("\n")
    print("Este script contiene funciones para procesar datos ENAHO.")
    print("Para ejecutar el pipeline completo, necesitas:")
    print("  1. DataFrame de individuos (módulo persona)")
    print("  2. DataFrame de hogares (módulo sumaria)")
    print("  3. (Opcional) Rutas a tablas CIIU y CNO")
    print("\n")
    print("Ejemplo de uso:")
    print("  df_final = pipeline_completo(df_individuos, df_hogares_enaho)")
    print("  resultados = analisis_descriptivo_ponderado(df_final, 'es_pobre_monetario')")
    print("\n")


# ================================================================================
# PUNTO DE ENTRADA
# ================================================================================

if __name__ == '__main__':
    main()
