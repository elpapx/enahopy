"""
ENAHO Merger - Geographic Merger Module
========================================

Specialized merger for integrating geographic information with ENAHO data.
Handles UBIGEO-based joins, territorial validation, and geographic enrichment.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from ..config import GeoMergeConfiguration


class GeographicMerger:
    """Specialized merger for geographic data integration with ENAHO surveys.


    Provides lightweight geographic merging capabilities for adding administrative
    division information (departamento, provincia, distrito) and geographic
    coordinates to ENAHO survey data using UBIGEO codes as merge keys.

    This class serves as a simplified geographic merger with basic functionality.
    For advanced features including validation, duplicate handling strategies,
    and quality assessment, use :class:`~enahopy.merger.ENAHOGeoMerger` instead.

    Attributes:
        config (GeoMergeConfiguration): Configuration controlling merge behavior
            including column names, merge semantics, and error handling.
        logger (logging.Logger): Logger instance for operation tracking and
            diagnostic information.

    Examples:
        Basic geographic merge:

        >>> from enahopy.merger.geographic.merger import GeographicMerger
        >>> import pandas as pd
        >>>
        >>> df_data = pd.DataFrame({
        ...     'ubigeo': ['150101', '150102'],
        ...     'ingreso': [2000, 1500]
        ... })
        >>> df_geo = pd.DataFrame({
        ...     'ubigeo': ['150101', '150102'],
        ...     'departamento': ['Lima', 'Lima'],
        ...     'provincia': ['Lima', 'Lima']
        ... })
        >>>
        >>> merger = GeographicMerger()
        >>> result, report = merger.merge(df_data, df_geo)
        >>> print(f"Records: {result.shape[0]}, Match rate: {report['match_rate']:.1f}%")
        Records: 2, Match rate: 100.0%

        With custom configuration:

        >>> from enahopy.merger.config import GeoMergeConfiguration
        >>> config = GeoMergeConfiguration(columna_union='cod_ubigeo')
        >>> merger = GeographicMerger(config=config)

    Note:
        - This is a simplified merger for basic use cases
        - For production use with validation, use ENAHOGeoMerger
        - Performs left join: all records from df_principal are preserved
        - No automatic duplicate handling or validation

    See Also:
        - :class:`~enahopy.merger.ENAHOGeoMerger`: Full-featured geographic merger
        - :class:`~enahopy.merger.config.GeoMergeConfiguration`: Configuration options
    """

    def __init__(
        self,
        config: Optional[GeoMergeConfiguration] = None,
        logger: Optional[logging.Logger] = None,
    ):
        """Initialize the geographic merger.

        Args:
            config: Optional configuration for merge operations. If None, uses
                default configuration with 'ubigeo' as merge column and left
                join semantics. Defaults to None.
            logger: Optional logger instance. If None, creates a new logger
                with module name. Defaults to None.

        Examples:
            Default initialization:

            >>> merger = GeographicMerger()

            With custom config:

            >>> from enahopy.merger.config import GeoMergeConfiguration
            >>> config = GeoMergeConfiguration(
            ...     columna_union='codigo_ubigeo',
            ...     validar_formato_ubigeo=False
            ... )
            >>> merger = GeographicMerger(config=config)
        """
        self.config = config or GeoMergeConfiguration()
        self.logger = logger or logging.getLogger(__name__)

    def merge(
        self,
        df_principal: pd.DataFrame,
        df_geografia: pd.DataFrame,
        columna_union: Optional[str] = None,
        columnas_geograficas: Optional[List[str]] = None,
        validate: bool = True,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Merge principal DataFrame with geographic information using UBIGEO.

        Performs a left join between the principal dataset and geographic
        reference data, adding administrative division columns. This is a
        simplified merge operation without advanced validation or duplicate
        handling.

        Args:
            df_principal: Principal DataFrame to enrich with geographic data.
                Must contain the merge key column (UBIGEO). All records from
                this DataFrame are preserved in the result (left join semantics).
            df_geografia: Geographic reference DataFrame containing UBIGEO codes
                and associated geographic information such as departamento,
                provincia, distrito, and optionally coordinates. Should ideally
                have unique UBIGEOs to avoid duplicate record creation.
            columna_union: Name of the column to use as merge key. Must exist
                in both DataFrames. If None, uses value from config.columna_union
                (default: "ubigeo"). Defaults to None.
            columnas_geograficas: Optional list of geographic column names to
                include from df_geografia. If None, all columns from df_geografia
                are included. This parameter is accepted for API compatibility
                but not currently used. Defaults to None.
            validate: If True, performs basic validation before merge. Currently
                not implemented - parameter accepted for future compatibility.
                Defaults to True.

        Returns:
            Tuple containing two elements:

            1. merged_df (pd.DataFrame): Result DataFrame combining df_principal
               with geographic columns from df_geografia. Preserves all records
               from df_principal (left join). Records without geographic match
               have NaN values in geographic columns.

            2. report (Dict[str, Any]): Basic merge report including:
               - input_rows (int): Number of records in df_principal
               - geography_rows (int): Number of records in df_geografia
               - output_rows (int): Number of records in merged result
               - match_rate (float): Currently always 100.0 (placeholder)

        Raises:
            GeoMergeError: If merge operation fails due to missing columns,
                incompatible data types, or pandas merge errors.

        Examples:
            Basic usage:

            >>> from enahopy.merger.geographic.merger import GeographicMerger
            >>> import pandas as pd
            >>>
            >>> df_data = pd.DataFrame({
            ...     'ubigeo': ['150101', '150102', '150103'],
            ...     'conglome': ['001', '002', '003'],
            ...     'ingreso': [2000, 1500, 1800]
            ... })
            >>> df_geo = pd.DataFrame({
            ...     'ubigeo': ['150101', '150102', '150103'],
            ...     'departamento': ['Lima', 'Lima', 'Lima'],
            ...     'provincia': ['Lima', 'Lima', 'Lima'],
            ...     'distrito': ['Lima', 'San Isidro', 'Miraflores']
            ... })
            >>>
            >>> merger = GeographicMerger()
            >>> result, report = merger.merge(df_data, df_geo)
            >>> print(f"Records: {report['output_rows']}")
            >>> print(result.columns.tolist())
            Records: 3
            ['ubigeo', 'conglome', 'ingreso', 'departamento', 'provincia', 'distrito']

            Custom merge column:

            >>> result, report = merger.merge(
            ...     df_data,
            ...     df_geo,
            ...     columna_union='ubigeo'
            ... )

            Handling partial matches:

            >>> df_partial_geo = pd.DataFrame({
            ...     'ubigeo': ['150101', '150102'],  # Missing 150103
            ...     'departamento': ['Lima', 'Lima']
            ... })
            >>> result, report = merger.merge(df_data, df_partial_geo)
            >>> print(result[result['departamento'].isna()])
            # Record with ubigeo='150103' will have NaN in departamento

        Note:
            - Uses pandas left join: all df_principal records are preserved
            - No automatic duplicate handling if df_geografia has duplicate UBIGEOs
            - No UBIGEO format validation or territorial consistency checks
            - Match rate in report is placeholder - use ENAHOGeoMerger for accurate metrics
            - For production use with validation, use :class:`~enahopy.merger.ENAHOGeoMerger`

        See Also:
            - :class:`~enahopy.merger.ENAHOGeoMerger`: Full-featured merger with validation
            - :meth:`~enahopy.merger.ENAHOGeoMerger.merge_geographic_data`: Advanced merge
        """
        columna_union = columna_union or self.config.columna_union

        self.logger.info(f"Iniciando fusión geográfica usando columna '{columna_union}'")

        # Realizar merge
        df_merged = pd.merge(df_principal, df_geografia, on=columna_union, how="left")

        # Generar reporte
        report = {
            "input_rows": len(df_principal),
            "geography_rows": len(df_geografia),
            "output_rows": len(df_merged),
            "match_rate": 100.0,
        }

        return df_merged, report


# Alias para compatibilidad
ENAHOGeographicMerger = GeographicMerger
