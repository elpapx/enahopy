"""
ENAHO Streaming Processing System - MEASURE Phase
=================================================

Advanced streaming system for processing large ENAHO datasets with minimal memory footprint.
Supports various file formats, parallel processing, and intelligent data pipeline optimization.
"""

import asyncio
import gc
import logging
import multiprocessing as mp
import tempfile
import threading
import time
from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Queue
from typing import Any, Callable, Dict, Generator, Iterator, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

try:
    import dask.dataframe as dd
    from dask.distributed import Client
    from dask.distributed import as_completed as dask_as_completed

    DASK_AVAILABLE = True
except ImportError:
    DASK_AVAILABLE = False

try:
    import pyarrow as pa
    import pyarrow.csv as pa_csv
    import pyarrow.parquet as pq

    ARROW_AVAILABLE = True
except ImportError:
    ARROW_AVAILABLE = False

try:
    import polars as pl

    POLARS_AVAILABLE = True
except ImportError:
    POLARS_AVAILABLE = False

from .memory_optimizer import DataFrameOptimizer, MemoryMonitor, memory_optimized_context


@dataclass
class StreamingConfig:
    """Configuration for streaming operations"""

    chunk_size: int = 10000
    max_memory_mb: float = 1000.0
    buffer_size: int = 3
    parallel_workers: int = None
    use_threading: bool = True
    use_arrow: bool = True
    temp_dir: Optional[Path] = None
    compression: str = "snappy"

    def __post_init__(self):
        if self.parallel_workers is None:
            self.parallel_workers = max(1, mp.cpu_count() - 1)


@dataclass
class ProcessingStats:
    """Statistics for streaming processing operations"""

    total_rows: int = 0
    chunks_processed: int = 0
    processing_time: float = 0.0
    peak_memory_mb: float = 0.0
    throughput_rows_per_sec: float = 0.0
    errors: List[str] = None

    def __post_init__(self):
        if self.errors is None:
            self.errors = []


class StreamingReader(ABC):
    """Abstract base class for streaming file readers"""

    @abstractmethod
    def read_chunks(self, chunk_size: int) -> Generator[pd.DataFrame, None, None]:
        """Read file in chunks"""
        pass

    @abstractmethod
    def get_total_rows(self) -> Optional[int]:
        """Get total number of rows if available"""
        pass

    @abstractmethod
    def get_column_info(self) -> Dict[str, Any]:
        """Get column information"""
        pass


class CSVStreamingReader(StreamingReader):
    """Streaming reader for CSV files"""

    def __init__(self, file_path: Path, **kwargs):
        self.file_path = file_path
        self.read_kwargs = kwargs
        self._total_rows = None
        self._columns = None

    def read_chunks(self, chunk_size: int) -> Generator[pd.DataFrame, None, None]:
        """Read CSV file in chunks"""
        try:
            reader = pd.read_csv(self.file_path, chunksize=chunk_size, **self.read_kwargs)
            for chunk in reader:
                if self._columns is None:
                    self._columns = chunk.columns.tolist()
                yield chunk
        except Exception as e:
            raise RuntimeError(f"Error reading CSV file {self.file_path}: {str(e)}")

    def get_total_rows(self) -> Optional[int]:
        """Estimate total rows by counting lines"""
        if self._total_rows is None:
            try:
                with open(self.file_path, "r", encoding="utf-8") as f:
                    self._total_rows = sum(1 for _ in f) - 1  # Subtract header
            except:
                self._total_rows = None
        return self._total_rows

    def get_column_info(self) -> Dict[str, Any]:
        """Get column information"""
        if self._columns is None:
            # Read first chunk to get columns
            first_chunk = next(self.read_chunks(1000))
            self._columns = first_chunk.columns.tolist()

        return {"columns": self._columns, "count": len(self._columns)}


class ParquetStreamingReader(StreamingReader):
    """Streaming reader for Parquet files"""

    def __init__(self, file_path: Path, **kwargs):
        self.file_path = file_path
        self.read_kwargs = kwargs
        self._parquet_file = None

        if not ARROW_AVAILABLE:
            raise ImportError("PyArrow is required for Parquet streaming")

    def read_chunks(self, chunk_size: int) -> Generator[pd.DataFrame, None, None]:
        """Read Parquet file in chunks"""
        try:
            parquet_file = pq.ParquetFile(self.file_path)

            for batch in parquet_file.iter_batches(batch_size=chunk_size):
                chunk = batch.to_pandas()
                yield chunk

        except Exception as e:
            raise RuntimeError(f"Error reading Parquet file {self.file_path}: {str(e)}")

    def get_total_rows(self) -> Optional[int]:
        """Get total number of rows from Parquet metadata"""
        try:
            parquet_file = pq.ParquetFile(self.file_path)
            return parquet_file.metadata.num_rows
        except:
            return None

    def get_column_info(self) -> Dict[str, Any]:
        """Get column information from Parquet schema"""
        try:
            parquet_file = pq.ParquetFile(self.file_path)
            schema = parquet_file.schema

            return {"columns": schema.names, "count": len(schema.names), "schema": schema}
        except:
            return {"columns": [], "count": 0}


class StreamingProcessor:
    """High-performance streaming processor for large datasets"""

    def __init__(
        self, config: Optional[StreamingConfig] = None, logger: Optional[logging.Logger] = None
    ):
        self.config = config or StreamingConfig()
        self.logger = logger or logging.getLogger(__name__)
        self.memory_monitor = MemoryMonitor(logger)
        self.optimizer = DataFrameOptimizer(logger)

        # Setup temp directory
        if self.config.temp_dir is None:
            self.config.temp_dir = Path(tempfile.gettempdir()) / "enaho_streaming"
        self.config.temp_dir.mkdir(exist_ok=True)

    def create_reader(self, file_path: Path, file_format: str = "auto") -> StreamingReader:
        """
        Create appropriate streaming reader for file

        Args:
            file_path: Path to file
            file_format: File format ('csv', 'parquet', 'auto')

        Returns:
            Streaming reader instance
        """
        if file_format == "auto":
            suffix = file_path.suffix.lower()
            if suffix == ".csv":
                file_format = "csv"
            elif suffix == ".parquet":
                file_format = "parquet"
            else:
                # Default to CSV
                file_format = "csv"

        if file_format == "csv":
            return CSVStreamingReader(file_path)
        elif file_format == "parquet":
            return ParquetStreamingReader(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_format}")

    def process_streaming(
        self,
        reader: StreamingReader,
        processing_function: Callable[[pd.DataFrame], pd.DataFrame],
        output_path: Optional[Path] = None,
        save_format: str = "parquet",
    ) -> ProcessingStats:
        """
        Process data using streaming with memory optimization

        Args:
            reader: Streaming reader instance
            processing_function: Function to apply to each chunk
            output_path: Optional output file path
            save_format: Output format ('parquet', 'csv')

        Returns:
            Processing statistics
        """
        stats = ProcessingStats()
        start_time = time.time()

        # Start memory monitoring
        self.memory_monitor.start_monitoring()

        try:
            with memory_optimized_context():
                first_chunk = True
                writer = None

                for chunk_num, chunk in enumerate(reader.read_chunks(self.config.chunk_size)):
                    chunk_start = time.time()

                    try:
                        # Optimize chunk memory usage
                        chunk = self.optimizer.optimize_dtypes(chunk)

                        # Apply processing function
                        processed_chunk = processing_function(chunk)

                        # Save processed chunk
                        if output_path:
                            if save_format == "parquet" and ARROW_AVAILABLE:
                                # Use Parquet for efficient storage
                                if first_chunk:
                                    # Create or overwrite file
                                    table = pa.Table.from_pandas(processed_chunk)
                                    writer = pq.ParquetWriter(output_path, table.schema)
                                    writer.write_table(table)
                                else:
                                    # Append to existing file
                                    table = pa.Table.from_pandas(processed_chunk)
                                    writer.write_table(table)
                            else:
                                # Fallback to CSV
                                mode = "w" if first_chunk else "a"
                                header = first_chunk
                                processed_chunk.to_csv(
                                    output_path, mode=mode, header=header, index=False
                                )

                        # Update statistics
                        chunk_rows = len(chunk)
                        stats.total_rows += chunk_rows
                        stats.chunks_processed += 1
                        first_chunk = False

                        chunk_time = time.time() - chunk_start
                        self.logger.debug(
                            f"Processed chunk {chunk_num + 1} ({chunk_rows} rows) in {chunk_time:.2f}s"
                        )

                        # Clean up memory
                        del chunk, processed_chunk

                        # Periodic garbage collection
                        if chunk_num % 10 == 0:
                            gc.collect()

                    except Exception as e:
                        error_msg = f"Error processing chunk {chunk_num + 1}: {str(e)}"
                        stats.errors.append(error_msg)
                        self.logger.error(error_msg)

                # Close writer if used
                if writer:
                    writer.close()

        except Exception as e:
            error_msg = f"Error in streaming processing: {str(e)}"
            stats.errors.append(error_msg)
            self.logger.error(error_msg)

        finally:
            # Stop monitoring and collect stats
            self.memory_monitor.stop_monitoring()
            stats.processing_time = time.time() - start_time
            stats.peak_memory_mb = self.memory_monitor.get_peak_memory()

            if stats.processing_time > 0:
                stats.throughput_rows_per_sec = stats.total_rows / stats.processing_time

        return stats

    def parallel_streaming(
        self,
        readers: List[StreamingReader],
        processing_function: Callable[[pd.DataFrame], pd.DataFrame],
        output_dir: Path,
        filename_template: str = "output_{index}.parquet",
    ) -> List[ProcessingStats]:
        """
        Process multiple files in parallel with streaming

        Args:
            readers: List of streaming readers
            processing_function: Function to apply to each chunk
            output_dir: Output directory
            filename_template: Template for output filenames

        Returns:
            List of processing statistics for each file
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        all_stats = []

        def process_single_file(index: int, reader: StreamingReader) -> ProcessingStats:
            """Process single file"""
            output_path = output_dir / filename_template.format(index=index)
            return self.process_streaming(reader, processing_function, output_path)

        if self.config.use_threading:
            # Use ThreadPoolExecutor for I/O bound operations
            with ThreadPoolExecutor(max_workers=self.config.parallel_workers) as executor:
                futures = [
                    executor.submit(process_single_file, i, reader)
                    for i, reader in enumerate(readers)
                ]

                for future in as_completed(futures):
                    try:
                        stats = future.result()
                        all_stats.append(stats)
                    except Exception as e:
                        error_stats = ProcessingStats()
                        error_stats.errors.append(f"Parallel processing error: {str(e)}")
                        all_stats.append(error_stats)
        else:
            # Use ProcessPoolExecutor for CPU-bound operations
            with ProcessPoolExecutor(max_workers=self.config.parallel_workers) as executor:
                futures = [
                    executor.submit(process_single_file, i, reader)
                    for i, reader in enumerate(readers)
                ]

                for future in as_completed(futures):
                    try:
                        stats = future.result()
                        all_stats.append(stats)
                    except Exception as e:
                        error_stats = ProcessingStats()
                        error_stats.errors.append(f"Parallel processing error: {str(e)}")
                        all_stats.append(error_stats)

        return all_stats

    def streaming_aggregation(
        self,
        reader: StreamingReader,
        group_columns: List[str],
        agg_functions: Dict[str, Union[str, List[str]]],
        output_path: Optional[Path] = None,
    ) -> pd.DataFrame:
        """
        Perform aggregation on streaming data with memory optimization

        Args:
            reader: Streaming reader
            group_columns: Columns to group by
            agg_functions: Aggregation functions
            output_path: Optional output path

        Returns:
            Aggregated DataFrame
        """
        self.logger.info("Starting streaming aggregation")

        # Use incremental aggregation with temporary storage
        temp_files = []
        chunk_results = []

        try:
            with memory_optimized_context():
                for chunk_num, chunk in enumerate(reader.read_chunks(self.config.chunk_size)):
                    # Optimize chunk
                    chunk = self.optimizer.optimize_dtypes(chunk)

                    # Perform aggregation on chunk
                    chunk_agg = chunk.groupby(group_columns).agg(agg_functions)

                    # Save chunk result to temporary file
                    temp_file = self.config.temp_dir / f"agg_chunk_{chunk_num}.parquet"

                    if ARROW_AVAILABLE:
                        chunk_agg.to_parquet(temp_file)
                    else:
                        chunk_agg.to_csv(temp_file)

                    temp_files.append(temp_file)

                    # Clean up
                    del chunk, chunk_agg
                    gc.collect()

                    self.logger.debug(f"Processed aggregation chunk {chunk_num + 1}")

                # Combine all chunk results
                self.logger.info(f"Combining {len(temp_files)} aggregation chunks")

                if ARROW_AVAILABLE and temp_files:
                    # Use Arrow for efficient reading and combining
                    combined_chunks = []
                    for temp_file in temp_files:
                        chunk_result = pd.read_parquet(temp_file)
                        combined_chunks.append(chunk_result)

                    # Combine and re-aggregate
                    if combined_chunks:
                        combined_df = pd.concat(combined_chunks, ignore_index=True)
                        final_result = combined_df.groupby(group_columns).agg(agg_functions)
                    else:
                        final_result = pd.DataFrame()
                else:
                    # Fallback to CSV
                    combined_chunks = []
                    for temp_file in temp_files:
                        chunk_result = pd.read_csv(temp_file, index_col=0)
                        combined_chunks.append(chunk_result)

                    if combined_chunks:
                        combined_df = pd.concat(combined_chunks, ignore_index=True)
                        final_result = combined_df.groupby(group_columns).agg(agg_functions)
                    else:
                        final_result = pd.DataFrame()

        finally:
            # Clean up temporary files
            for temp_file in temp_files:
                try:
                    temp_file.unlink()
                except:
                    pass

        # Save final result if requested
        if output_path and not final_result.empty:
            if output_path.suffix == ".parquet" and ARROW_AVAILABLE:
                final_result.to_parquet(output_path)
            else:
                final_result.to_csv(output_path)

        self.logger.info(f"Streaming aggregation completed: {len(final_result)} aggregated groups")
        return final_result

    def streaming_join(
        self,
        left_reader: StreamingReader,
        right_df: pd.DataFrame,
        on: Union[str, List[str]],
        how: str = "inner",
        output_path: Optional[Path] = None,
    ) -> ProcessingStats:
        """
        Perform streaming join between large dataset and smaller DataFrame

        Args:
            left_reader: Streaming reader for large dataset
            right_df: Smaller DataFrame to join with
            on: Column(s) to join on
            how: Join type
            output_path: Optional output path

        Returns:
            Processing statistics
        """
        stats = ProcessingStats()
        start_time = time.time()

        self.memory_monitor.start_monitoring()

        try:
            with memory_optimized_context():
                first_chunk = True
                writer = None

                # Optimize right DataFrame
                right_df = self.optimizer.optimize_dtypes(right_df)

                for chunk_num, chunk in enumerate(left_reader.read_chunks(self.config.chunk_size)):
                    # Optimize chunk
                    chunk = self.optimizer.optimize_dtypes(chunk)

                    # Perform join
                    joined_chunk = chunk.merge(right_df, on=on, how=how)

                    # Save result
                    if output_path:
                        if output_path.suffix == ".parquet" and ARROW_AVAILABLE:
                            if first_chunk:
                                table = pa.Table.from_pandas(joined_chunk)
                                writer = pq.ParquetWriter(output_path, table.schema)
                                writer.write_table(table)
                            else:
                                table = pa.Table.from_pandas(joined_chunk)
                                writer.write_table(table)
                        else:
                            mode = "w" if first_chunk else "a"
                            header = first_chunk
                            joined_chunk.to_csv(output_path, mode=mode, header=header, index=False)

                    stats.total_rows += len(joined_chunk)
                    stats.chunks_processed += 1
                    first_chunk = False

                    # Clean up
                    del chunk, joined_chunk
                    gc.collect()

                if writer:
                    writer.close()

        except Exception as e:
            error_msg = f"Error in streaming join: {str(e)}"
            stats.errors.append(error_msg)
            self.logger.error(error_msg)

        finally:
            self.memory_monitor.stop_monitoring()
            stats.processing_time = time.time() - start_time
            stats.peak_memory_mb = self.memory_monitor.get_peak_memory()

            if stats.processing_time > 0:
                stats.throughput_rows_per_sec = stats.total_rows / stats.processing_time

        return stats


class DaskStreamingProcessor:
    """Dask-based streaming processor for distributed computing"""

    def __init__(
        self,
        config: Optional[StreamingConfig] = None,
        logger: Optional[logging.Logger] = None,
        client: Optional["Client"] = None,
    ):
        self.config = config or StreamingConfig()
        self.logger = logger or logging.getLogger(__name__)
        self.client = client

        if not DASK_AVAILABLE:
            raise ImportError("Dask is required for distributed streaming")

    def process_with_dask(
        self,
        file_path: Path,
        processing_function: Callable[[pd.DataFrame], pd.DataFrame],
        output_path: Path,
    ) -> ProcessingStats:
        """
        Process large file using Dask for distributed computing

        Args:
            file_path: Input file path
            processing_function: Processing function
            output_path: Output file path

        Returns:
            Processing statistics
        """
        start_time = time.time()
        stats = ProcessingStats()

        try:
            # Read file as Dask DataFrame
            if file_path.suffix == ".parquet":
                ddf = dd.read_parquet(file_path)
            else:
                ddf = dd.read_csv(file_path)

            # Apply processing function
            processed_ddf = ddf.map_partitions(processing_function)

            # Write result
            if output_path.suffix == ".parquet":
                processed_ddf.to_parquet(output_path, compression=self.config.compression)
            else:
                processed_ddf.to_csv(output_path, index=False)

            stats.total_rows = len(ddf)
            stats.chunks_processed = ddf.npartitions

        except Exception as e:
            error_msg = f"Error in Dask processing: {str(e)}"
            stats.errors.append(error_msg)
            self.logger.error(error_msg)

        finally:
            stats.processing_time = time.time() - start_time
            if stats.processing_time > 0:
                stats.throughput_rows_per_sec = stats.total_rows / stats.processing_time

        return stats


def create_streaming_processor(
    config: Optional[StreamingConfig] = None,
    logger: Optional[logging.Logger] = None,
    use_dask: bool = False,
) -> Union[StreamingProcessor, DaskStreamingProcessor]:
    """
    Factory function to create streaming processor

    Args:
        config: Streaming configuration
        logger: Logger instance
        use_dask: Whether to use Dask for distributed processing

    Returns:
        Streaming processor instance
    """
    if use_dask and DASK_AVAILABLE:
        return DaskStreamingProcessor(config, logger)
    else:
        return StreamingProcessor(config, logger)


__all__ = [
    "StreamingConfig",
    "ProcessingStats",
    "StreamingReader",
    "CSVStreamingReader",
    "ParquetStreamingReader",
    "StreamingProcessor",
    "DaskStreamingProcessor",
    "create_streaming_processor",
]
