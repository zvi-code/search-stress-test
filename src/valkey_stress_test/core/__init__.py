# src/valkey_stress_test/core/__init__.py
"""Core components for valkey stress testing."""

from .dataset import Dataset, DatasetInfo, DatasetManager
from .vector_ops import (
    VectorOperations, 
    VectorSampler, 
    VectorExpansionConfig,
    calculate_recall,
    generate_test_vectors
)
from .connection import (
    ConnectionManager, 
    AsyncRedisPool,
    ConnectionConfig,
    ClientPool
)
from .metrics import (
    MetricCollector, 
    MetricAggregator,
    MemoryMetrics,
    PerformanceMetrics,
    OperationMetrics
)
from .config import (
    Config, 
    ConfigValidator,
    IndexConfig,
    WorkloadConfig,
    RedisConfig,
    MonitoringConfig,
    OutputConfig
)

__all__ = [
    # Dataset
    "Dataset",
    "DatasetInfo", 
    "DatasetManager",
    
    # Vector operations
    "VectorOperations",
    "VectorSampler",
    "VectorExpansionConfig",
    "calculate_recall",
    "generate_test_vectors",
    
    # Connection management
    "ConnectionManager",
    "AsyncRedisPool",
    "ConnectionConfig",
    "ClientPool",
    
    # Metrics
    "MetricCollector",
    "MetricAggregator",
    "MemoryMetrics",
    "PerformanceMetrics",
    "OperationMetrics",
    
    # Configuration
    "Config",
    "ConfigValidator",
    "IndexConfig",
    "WorkloadConfig",
    "RedisConfig",
    "MonitoringConfig",
    "OutputConfig",
]