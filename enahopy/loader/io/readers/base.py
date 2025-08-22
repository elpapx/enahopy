"""
ENAHO Readers Base Module
========================

Implementación base para lectores de archivos.
Contiene funcionalidad común compartida entre todos los readers.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from ...io.base import IReader


class BaseReader(IReader):
    """Clase base con funcionalidad común para metadatos."""

    def _extract_base_metadata(self) -> Dict:
        """Extrae metadatos básicos del archivo"""
        file_stat = self.file_path.stat()
        return {
            "file_info": {
                "file_path": str(self.file_path),
                "file_size_mb": file_stat.st_size / (1024 * 1024),
                "creation_time": datetime.fromtimestamp(file_stat.st_ctime).isoformat(),
                "modification_time": datetime.fromtimestamp(file_stat.st_mtime).isoformat(),
            },
            "dataset_info": {},
            "variables": {},
            "value_labels": {},
        }

    def _populate_spss_dta_metadata(self, metadata: Dict, meta_obj) -> Dict:
        """Popula el diccionario de metadatos con información común de SPSS y Stata."""
        metadata["dataset_info"].update(
            {
                "number_rows": meta_obj.number_rows,
                "number_columns": meta_obj.number_columns,
                "file_encoding": meta_obj.file_encoding,
                "dataset_label": getattr(meta_obj, "table_name", ""),
            }
        )
        metadata["variables"].update(
            {
                "column_names": meta_obj.column_names,
                "column_labels": meta_obj.column_labels,
                "readstat_variable_types": getattr(meta_obj, "readstat_variable_types", {}),
                "variable_format": getattr(meta_obj, "variable_format", {}),
            }
        )
        metadata["value_labels"].update(
            {
                "value_labels": getattr(meta_obj, "value_labels", {}),
                "variable_value_labels": getattr(meta_obj, "variable_value_labels", {}),
            }
        )
        return metadata


__all__ = ["BaseReader"]
