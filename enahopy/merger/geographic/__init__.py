"""
ENAHO Merger - Geographic Module
================================

Módulo para fusión con datos geográficos.

This module provides specialized geographic merge capabilities for enriching ENAHO
survey data with administrative division information and geographic coordinates
using UBIGEO codes (Peruvian geographic identifiers).

Main Components:
    - GeographicMerger: Lightweight geographic merge operations
    - ENAHOGeographicMerger: Alias for GeographicMerger (compatibility)

Key Features:
    - UBIGEO-based geographic data integration
    - Administrative division enrichment (departamento, provincia, distrito)
    - Left join semantics preserving all principal records
    - Configurable merge column names
    - Basic merge reporting with match rates

The Geographic Merger provides simplified geographic integration with minimal
configuration. For advanced features including validation, duplicate handling,
and quality assessment, use the full ENAHOGeoMerger from the parent module.

Examples:
    Basic geographic merge:

    >>> from enahopy.merger.geographic import GeographicMerger
    >>> merger = GeographicMerger()
    >>> result, report = merger.merge(
    ...     df_principal=df_data,
    ...     df_geografia=df_geo
    ... )
    >>> print(f"Match rate: {report['match_rate']:.1f}%")

    With custom configuration:

    >>> from enahopy.merger.config import GeoMergeConfiguration
    >>> config = GeoMergeConfiguration(columna_union='cod_ubigeo')
    >>> merger = GeographicMerger(config=config)

Note:
    This is a simplified merger for basic use cases. For production environments
    with validation requirements, use ENAHOGeoMerger from the parent module which
    provides comprehensive validation, duplicate handling strategies, territorial
    consistency checks, and detailed quality reporting.

See Also:
    - :class:`~enahopy.merger.ENAHOGeoMerger`: Full-featured geographic merger
    - :class:`~enahopy.merger.config.GeoMergeConfiguration`: Configuration options
    - :mod:`~enahopy.merger`: Parent merger module
"""

from .merger import ENAHOGeographicMerger, GeographicMerger

__all__ = ["GeographicMerger", "ENAHOGeographicMerger"]
