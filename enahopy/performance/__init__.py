"""
ENAHO Performance Package - MEASURE Phase
==========================================

High-performance optimization suite for ENAHO data processing including:
- Async downloading with aiohttp
- Memory profiling and optimization  
- Streaming processing for large datasets
- Performance benchmarking and monitoring
"""

# Memory optimization
try:
    from .memory_optimizer import DataFrameOptimizer, MemoryMonitor, MemoryProfile, MemorySnapshot
    from .memory_optimizer import StreamingProcessor as MemoryStreamingProcessor
    from .memory_optimizer import (
        create_memory_optimizer,
        memory_optimized_context,
        memory_profile_decorator,
        optimize_pandas_settings,
    )

    MEMORY_OPTIMIZER_AVAILABLE = True
except ImportError:
    MEMORY_OPTIMIZER_AVAILABLE = False
    MemoryMonitor = None
    DataFrameOptimizer = None
    MemoryStreamingProcessor = None
    MemoryProfile = None
    MemorySnapshot = None
    memory_profile_decorator = None
    memory_optimized_context = None
    optimize_pandas_settings = None
    create_memory_optimizer = None

# Streaming processing
try:
    from .streaming import (
        CSVStreamingReader,
        DaskStreamingProcessor,
        ParquetStreamingReader,
        ProcessingStats,
        StreamingConfig,
        StreamingProcessor,
        StreamingReader,
        create_streaming_processor,
    )

    STREAMING_AVAILABLE = True
except ImportError:
    STREAMING_AVAILABLE = False
    StreamingConfig = None
    ProcessingStats = None
    StreamingReader = None
    CSVStreamingReader = None
    ParquetStreamingReader = None
    StreamingProcessor = None
    DaskStreamingProcessor = None
    create_streaming_processor = None

# Performance benchmarking
try:
    from .benchmarks import (
        BenchmarkResult,
        BenchmarkSuite,
        ENAHOBenchmarkSuite,
        PerformanceProfiler,
        SystemInfo,
        run_quick_benchmark,
    )

    BENCHMARKS_AVAILABLE = True
except ImportError:
    BENCHMARKS_AVAILABLE = False
    SystemInfo = None
    BenchmarkResult = None
    BenchmarkSuite = None
    PerformanceProfiler = None
    ENAHOBenchmarkSuite = None
    run_quick_benchmark = None

# Async downloading (requires special handling due to import from loader)
try:
    from ..loader.io.downloaders.async_downloader import (
        AsyncNetworkUtils,
        AsyncRetryHandler,
        ENAHOAsyncDownloader,
    )
    from ..loader.io.downloaders.async_downloader import (
        PerformanceMonitor as DownloadPerformanceMonitor,
    )
    from ..loader.io.downloaders.async_downloader import (
        StreamingDownloader,
        create_async_downloader,
    )

    ASYNC_DOWNLOADER_AVAILABLE = True
except ImportError:
    ASYNC_DOWNLOADER_AVAILABLE = False
    ENAHOAsyncDownloader = None
    AsyncNetworkUtils = None
    StreamingDownloader = None
    DownloadPerformanceMonitor = None
    AsyncRetryHandler = None
    create_async_downloader = None


def get_performance_status() -> dict:
    """Get status of all performance components"""
    return {
        "memory_optimizer": MEMORY_OPTIMIZER_AVAILABLE,
        "streaming": STREAMING_AVAILABLE,
        "benchmarks": BENCHMARKS_AVAILABLE,
        "async_downloader": ASYNC_DOWNLOADER_AVAILABLE,
    }


def show_performance_status():
    """Display performance components status"""
    status = get_performance_status()

    print("ENAHO Performance Components Status:")
    print("-" * 40)

    components = {
        "Memory Optimizer": status["memory_optimizer"],
        "Streaming Processing": status["streaming"],
        "Performance Benchmarks": status["benchmarks"],
        "Async Downloader": status["async_downloader"],
    }

    for name, available in components.items():
        symbol = "[OK]" if available else "[X]"
        status_text = "Available" if available else "Not Available"
        print(f"{symbol} {name}: {status_text}")

    # Show missing dependencies
    missing_deps = []
    if not ASYNC_DOWNLOADER_AVAILABLE:
        missing_deps.append("aiohttp, aiofiles")
    if not MEMORY_OPTIMIZER_AVAILABLE:
        missing_deps.append("psutil, memory-profiler")
    if not STREAMING_AVAILABLE:
        missing_deps.append("pyarrow (optional), polars (optional)")

    if missing_deps:
        print(f"\nMissing optional dependencies: {', '.join(missing_deps)}")
        print("Install with: pip install aiohttp aiofiles psutil memory-profiler pyarrow")


def create_performance_suite(logger=None):
    """
    Create complete performance optimization suite

    Args:
        logger: Optional logger instance

    Returns:
        Dictionary with all available performance tools
    """
    suite = {}

    if MEMORY_OPTIMIZER_AVAILABLE:
        suite.update(create_memory_optimizer(logger))

    if STREAMING_AVAILABLE:
        suite["streaming_processor"] = create_streaming_processor(logger=logger)

    if BENCHMARKS_AVAILABLE:
        suite["benchmark_suite"] = ENAHOBenchmarkSuite(logger)

    if ASYNC_DOWNLOADER_AVAILABLE:
        suite["async_downloader"] = create_async_downloader(logger=logger)

    return suite


# Quick performance optimization functions
def optimize_for_large_datasets():
    """Apply performance optimizations for large dataset processing"""
    if MEMORY_OPTIMIZER_AVAILABLE:
        optimize_pandas_settings()
        return True
    return False


def quick_performance_check(logger=None):
    """Run quick performance check and return recommendations"""
    if BENCHMARKS_AVAILABLE:
        return run_quick_benchmark(logger)
    else:
        return {
            "error": "Benchmarking not available",
            "recommendation": "Install performance dependencies: pip install psutil memory-profiler",
        }


# Export all available components
__all__ = []

if MEMORY_OPTIMIZER_AVAILABLE:
    __all__.extend(
        [
            "MemoryMonitor",
            "DataFrameOptimizer",
            "MemoryStreamingProcessor",
            "MemoryProfile",
            "MemorySnapshot",
            "memory_profile_decorator",
            "memory_optimized_context",
            "optimize_pandas_settings",
            "create_memory_optimizer",
        ]
    )

if STREAMING_AVAILABLE:
    __all__.extend(
        [
            "StreamingConfig",
            "ProcessingStats",
            "StreamingReader",
            "CSVStreamingReader",
            "ParquetStreamingReader",
            "StreamingProcessor",
            "DaskStreamingProcessor",
            "create_streaming_processor",
        ]
    )

if BENCHMARKS_AVAILABLE:
    __all__.extend(
        [
            "SystemInfo",
            "BenchmarkResult",
            "BenchmarkSuite",
            "PerformanceProfiler",
            "ENAHOBenchmarkSuite",
            "run_quick_benchmark",
        ]
    )

if ASYNC_DOWNLOADER_AVAILABLE:
    __all__.extend(
        [
            "ENAHOAsyncDownloader",
            "AsyncNetworkUtils",
            "StreamingDownloader",
            "DownloadPerformanceMonitor",
            "AsyncRetryHandler",
            "create_async_downloader",
        ]
    )

# Always available utility functions
__all__.extend(
    [
        "get_performance_status",
        "show_performance_status",
        "create_performance_suite",
        "optimize_for_large_datasets",
        "quick_performance_check",
    ]
)
