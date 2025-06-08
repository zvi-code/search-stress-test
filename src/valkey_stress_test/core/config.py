# src/valkey_stress_test/core/config.py
"""Configuration parsing and validation."""

from __future__ import annotations

import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field
import logging
import os

logger = logging.getLogger(__name__)


@dataclass
class IndexConfig:
    """Configuration for vector index."""
    algorithm: str = "HNSW"
    m: int = 16
    ef_construction: int = 356
    ef_runtime: int = 200
    distance_metric: str = "L2"
    dimensions: int = 1536
    initial_cap: int = 10000
    
    def validate(self) -> None:
        """Validate index configuration."""
        if self.algorithm not in ["HNSW", "FLAT"]:
            raise ValueError(f"Invalid algorithm: {self.algorithm}. Must be HNSW or FLAT")
        
        if self.distance_metric not in ["L2", "IP", "COSINE"]:
            raise ValueError(f"Invalid distance metric: {self.distance_metric}")
        
        if self.dimensions <= 0:
            raise ValueError(f"Dimensions must be positive: {self.dimensions}")
        
        if self.algorithm == "HNSW":
            if not 1 <= self.m <= 512:
                raise ValueError(f"M must be between 1 and 512: {self.m}")
            if not 1 <= self.ef_construction <= 4096:
                raise ValueError(f"ef_construction must be between 1 and 4096: {self.ef_construction}")
            if not 1 <= self.ef_runtime <= 4096:
                raise ValueError(f"ef_runtime must be between 1 and 4096: {self.ef_runtime}")


@dataclass
class WorkloadConfig:
    """Configuration for workload execution."""
    n_threads: int = 8
    n_clients: int = 1000
    batch_size: int = 1000
    operation_timeout: float = 30.0
    query_k: int = 10
    
    def validate(self) -> None:
        """Validate workload configuration."""
        if self.n_threads <= 0:
            raise ValueError(f"n_threads must be positive: {self.n_threads}")
        
        if self.n_clients <= 0:
            raise ValueError(f"n_clients must be positive: {self.n_clients}")
        
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive: {self.batch_size}")
        
        if self.operation_timeout <= 0:
            raise ValueError(f"operation_timeout must be positive: {self.operation_timeout}")
        
        if self.query_k <= 0:
            raise ValueError(f"query_k must be positive: {self.query_k}")


@dataclass
class RedisConfig:
    """Redis connection configuration."""
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: Optional[str] = None
    max_connections: int = 1000
    socket_timeout: float = 30.0
    socket_connect_timeout: float = 10.0
    
    def validate(self) -> None:
        """Validate Redis configuration."""
        if not self.host:
            raise ValueError("Redis host cannot be empty")
        
        if not 1 <= self.port <= 65535:
            raise ValueError(f"Invalid port number: {self.port}")
        
        if self.db < 0:
            raise ValueError(f"Database number must be non-negative: {self.db}")
        
        if self.max_connections <= 0:
            raise ValueError(f"max_connections must be positive: {self.max_connections}")


@dataclass
class MonitoringConfig:
    """Monitoring configuration."""
    sampling_interval: float = 10.0
    memory_metrics: List[str] = field(default_factory=lambda: [
        "rss_mb", "active_mb", "resident_mb", "allocated_mb", "fragmentation_ratio"
    ])
    export_format: str = "csv"
    prometheus_pushgateway: Optional[str] = None
    
    def validate(self) -> None:
        """Validate monitoring configuration."""
        if self.sampling_interval <= 0:
            raise ValueError(f"sampling_interval must be positive: {self.sampling_interval}")
        
        if self.export_format not in ["csv", "prometheus", "both"]:
            raise ValueError(f"Invalid export format: {self.export_format}")


@dataclass
class OutputConfig:
    """Output configuration."""
    csv_path: Path = Path("output/metrics.csv")
    summary_path: Path = Path("output/summary.csv")
    log_level: str = "INFO"
    
    def validate(self) -> None:
        """Validate output configuration."""
        # Create output directories if they don't exist
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.summary_path.parent.mkdir(parents=True, exist_ok=True)
        
        if self.log_level not in ["DEBUG", "INFO", "WARNING", "ERROR"]:
            raise ValueError(f"Invalid log level: {self.log_level}")


class Config:
    """Main configuration container."""
    
    def __init__(self, config_path: Optional[Path] = None):
        """Initialize configuration."""
        self.config_path = config_path
        self.data: Dict[str, Any] = {}
        
        # Initialize with defaults
        self.redis = RedisConfig()
        self.index = IndexConfig()
        self.workload = WorkloadConfig()
        self.monitoring = MonitoringConfig()
        self.output = OutputConfig()
        
        # Load from file if provided
        if config_path:
            self.load()
    
    def load(self) -> None:
        """Load configuration from file."""
        if not self.config_path:
            raise ValueError("No configuration path specified")
        
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        logger.info(f"Loading configuration from {self.config_path}")
        
        try:
            with open(self.config_path, 'r') as f:
                self.data = yaml.safe_load(f) or {}
            
            # Merge with environment variables
            self._merge_env_vars()
            
            # Update configuration objects
            self._update_from_dict()
            
            # Validate all configurations
            self.validate()
            
            logger.info("Configuration loaded successfully")
            
        except yaml.YAMLError as e:
            logger.error(f"Failed to parse YAML configuration: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise
    
    def _merge_env_vars(self) -> None:
        """Merge environment variables into configuration."""
        # Redis configuration from environment
        if "REDIS_HOST" in os.environ:
            self.data.setdefault("redis", {})["host"] = os.environ["REDIS_HOST"]
        if "REDIS_PORT" in os.environ:
            self.data.setdefault("redis", {})["port"] = int(os.environ["REDIS_PORT"])
        if "REDIS_PASSWORD" in os.environ:
            self.data.setdefault("redis", {})["password"] = os.environ["REDIS_PASSWORD"]
        
        # Output configuration from environment
        if "VST_OUTPUT_DIR" in os.environ:
            output_dir = Path(os.environ["VST_OUTPUT_DIR"])
            self.data.setdefault("output", {})["csv_path"] = str(output_dir / "metrics.csv")
            self.data.setdefault("output", {})["summary_path"] = str(output_dir / "summary.csv")
        
        if "VST_LOG_LEVEL" in os.environ:
            self.data.setdefault("output", {})["log_level"] = os.environ["VST_LOG_LEVEL"]
    
    def _update_from_dict(self) -> None:
        """Update configuration objects from loaded data."""
        # Update Redis config
        if "redis" in self.data:
            redis_data = self.data["redis"]
            self.redis = RedisConfig(
                host=redis_data.get("host", self.redis.host),
                port=redis_data.get("port", self.redis.port),
                db=redis_data.get("db", self.redis.db),
                password=redis_data.get("password", self.redis.password),
                max_connections=redis_data.get("max_connections", self.redis.max_connections),
                socket_timeout=redis_data.get("socket_timeout", self.redis.socket_timeout),
                socket_connect_timeout=redis_data.get("socket_connect_timeout", self.redis.socket_connect_timeout),
            )
        
        # Update Index config
        if "index" in self.data:
            index_data = self.data["index"]
            self.index = IndexConfig(
                algorithm=index_data.get("algorithm", self.index.algorithm),
                m=index_data.get("m", self.index.m),
                ef_construction=index_data.get("ef_construction", self.index.ef_construction),
                ef_runtime=index_data.get("ef_runtime", self.index.ef_runtime),
                distance_metric=index_data.get("distance_metric", self.index.distance_metric),
                dimensions=index_data.get("dimensions", self.index.dimensions),
                initial_cap=index_data.get("initial_cap", self.index.initial_cap),
            )
        
        # Update Workload config
        if "workload" in self.data:
            workload_data = self.data["workload"]
            self.workload = WorkloadConfig(
                n_threads=workload_data.get("n_threads", self.workload.n_threads),
                n_clients=workload_data.get("n_clients", self.workload.n_clients),
                batch_size=workload_data.get("batch_size", self.workload.batch_size),
                operation_timeout=workload_data.get("operation_timeout", self.workload.operation_timeout),
                query_k=workload_data.get("query_k", self.workload.query_k),
            )
        
        # Update Monitoring config
        if "monitoring" in self.data:
            monitoring_data = self.data["monitoring"]
            self.monitoring = MonitoringConfig(
                sampling_interval=monitoring_data.get("sampling_interval", self.monitoring.sampling_interval),
                memory_metrics=monitoring_data.get("memory_metrics", self.monitoring.memory_metrics),
                export_format=monitoring_data.get("export_format", self.monitoring.export_format),
                prometheus_pushgateway=monitoring_data.get("prometheus_pushgateway", self.monitoring.prometheus_pushgateway),
            )
        
        # Update Output config
        if "output" in self.data:
            output_data = self.data["output"]
            self.output = OutputConfig(
                csv_path=Path(output_data.get("csv_path", str(self.output.csv_path))),
                summary_path=Path(output_data.get("summary_path", str(self.output.summary_path))),
                log_level=output_data.get("log_level", self.output.log_level),
            )
    
    def validate(self) -> None:
        """Validate all configuration sections."""
        try:
            self.redis.validate()
            self.index.validate()
            self.workload.validate()
            self.monitoring.validate()
            self.output.validate()
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            raise
    
    def get_index_config(self) -> IndexConfig:
        """Get index configuration."""
        return self.index
        
    def get_workload_config(self) -> WorkloadConfig:
        """Get workload configuration."""
        return self.workload
    
    def get_redis_config(self) -> RedisConfig:
        """Get Redis configuration."""
        return self.redis
    
    def get_monitoring_config(self) -> MonitoringConfig:
        """Get monitoring configuration."""
        return self.monitoring
    
    def get_output_config(self) -> OutputConfig:
        """Get output configuration."""
        return self.output
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "redis": {
                "host": self.redis.host,
                "port": self.redis.port,
                "db": self.redis.db,
                "max_connections": self.redis.max_connections,
            },
            "index": {
                "algorithm": self.index.algorithm,
                "dimensions": self.index.dimensions,
                "distance_metric": self.index.distance_metric,
                "m": self.index.m,
                "ef_construction": self.index.ef_construction,
                "ef_runtime": self.index.ef_runtime,
            },
            "workload": {
                "n_threads": self.workload.n_threads,
                "n_clients": self.workload.n_clients,
                "batch_size": self.workload.batch_size,
                "operation_timeout": self.workload.operation_timeout,
                "query_k": self.workload.query_k,
            },
            "monitoring": {
                "sampling_interval": self.monitoring.sampling_interval,
                "memory_metrics": self.monitoring.memory_metrics,
                "export_format": self.monitoring.export_format,
            },
            "output": {
                "csv_path": str(self.output.csv_path),
                "summary_path": str(self.output.summary_path),
                "log_level": self.output.log_level,
            },
        }


class ConfigValidator:
    """Validates configuration values."""
    
    @staticmethod
    def validate(config: Dict[str, Any]) -> bool:
        """Validate configuration dictionary."""
        try:
            # Create a temporary Config object to validate
            temp_config = Config()
            temp_config.data = config
            temp_config._update_from_dict()
            temp_config.validate()
            return True
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            return False
    
    @staticmethod
    def validate_scenario_config(scenario_config: Dict[str, Any]) -> bool:
        """Validate scenario-specific configuration."""
        required_fields = ["name", "dataset", "steps"]
        
        # Check required fields
        for field in required_fields:
            if field not in scenario_config:
                logger.error(f"Missing required field in scenario: {field}")
                return False
        
        # Validate steps
        steps = scenario_config.get("steps", [])
        if not steps:
            logger.error("Scenario must have at least one step")
            return False
        
        for i, step in enumerate(steps):
            if "name" not in step:
                logger.error(f"Step {i} missing required field: name")
                return False
            if "type" not in step:
                logger.error(f"Step {i} missing required field: type")
                return False
            
            step_type = step["type"]
            if step_type not in ["workload", "wait", "checkpoint"]:
                logger.error(f"Step {i} has invalid type: {step_type}")
                return False
            
            if step_type == "workload" and "workload" not in step:
                logger.error(f"Step {i} of type 'workload' missing workload field")
                return False
        
        return True