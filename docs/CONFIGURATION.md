# Configuration Guide

This comprehensive guide covers all configuration options for the Valkey Memory Stress Testing tool, from basic settings to advanced tuning parameters.

## Table of Contents

- [Overview](#overview)
- [Configuration Files](#configuration-files)
- [Redis Configuration](#redis-configuration)
- [Index Configuration](#index-configuration)
- [Workload Configuration](#workload-configuration)
- [Monitoring Configuration](#monitoring-configuration)
- [Environment Variables](#environment-variables)
- [Configuration Validation](#configuration-validation)
- [Advanced Configuration](#advanced-configuration)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

## Overview

The Valkey Memory Stress Testing tool uses YAML configuration files to define:

- **Redis Connection Settings**: Host, port, authentication, connection pooling
- **Vector Index Parameters**: Algorithm, dimensions, distance metrics
- **Workload Settings**: Concurrency, batch sizes, timeouts
- **Monitoring Options**: Metrics collection, sampling intervals
- **Dataset Configurations**: Data sources and processing options

Configuration precedence (highest to lowest):
1. Command-line arguments (`--config`, `--redis-host`, etc.)
2. Environment variables (`VST_REDIS_HOST`, etc.)
3. Custom configuration file
4. Default configuration (`config/default.yaml`)

## Configuration Files

### Default Configuration

The tool ships with a default configuration at `config/default.yaml`:

```yaml
# Redis connection settings
redis:
  host: localhost
  port: 6379
  db: 0
  max_connections: 1000
  password: null
  ssl: false
  ssl_cert_reqs: required
  socket_timeout: 30.0
  socket_connect_timeout: 30.0
  retry_on_timeout: true
  health_check_interval: 30

# Vector index configuration
index:
  algorithm: HNSW
  dimensions: 1536
  distance_metric: L2
  m: 16
  ef_construction: 356
  ef_runtime: 200
  initial_cap: 10000
  block_size: 1024

# Workload execution settings  
workload:
  n_threads: 8
  n_clients_per_thread: 125
  batch_size: 1000
  operation_timeout: 30.0
  max_retries: 3
  retry_delay: 1.0
  jitter: true

# Monitoring and metrics
monitoring:
  sampling_interval: 10.0
  memory_metrics:
    - rss_mb
    - active_mb
    - resident_mb
    - allocated_mb
    - fragmentation_ratio
  performance_metrics:
    - operations_per_second
    - latency_p50
    - latency_p95
    - latency_p99
  collection_enabled: true
  export_format: json
```

### Custom Configuration

Create custom configurations for different environments:

```yaml
# config/production.yaml
redis:
  host: redis-cluster.example.com
  port: 6379
  password: "${REDIS_PASSWORD}"
  ssl: true
  max_connections: 2000

workload:
  n_threads: 16
  n_clients_per_thread: 250
  batch_size: 2000
  operation_timeout: 60.0

monitoring:
  sampling_interval: 5.0
  export_format: prometheus
```

```yaml
# config/development.yaml
redis:
  host: localhost
  port: 6379
  db: 1

workload:
  n_threads: 2
  n_clients_per_thread: 10
  batch_size: 100

monitoring:
  sampling_interval: 1.0
  collection_enabled: true
```

## Redis Configuration

### Basic Connection Settings

```yaml
redis:
  # Required settings
  host: localhost              # Redis server hostname
  port: 6379                  # Redis server port
  db: 0                       # Redis database number (0-15)
  
  # Optional authentication
  password: null              # Redis password (or use environment variable)
  username: null              # Redis username (Redis 6.0+)
  
  # SSL/TLS settings
  ssl: false                  # Enable SSL/TLS
  ssl_cert_reqs: required     # SSL certificate requirements
  ssl_ca_certs: null          # CA certificate file path
  ssl_cert_file: null         # Client certificate file path
  ssl_key_file: null          # Client private key file path
```

### Connection Pooling

```yaml
redis:
  # Connection pool settings
  max_connections: 1000       # Maximum connections in pool
  max_connections_per_pool: 50 # Max connections per pool instance
  retry_on_timeout: true      # Retry operations on timeout
  socket_timeout: 30.0        # Socket operation timeout (seconds)
  socket_connect_timeout: 30.0 # Socket connection timeout (seconds)
  socket_keepalive: true      # Enable TCP keepalive
  socket_keepalive_options: {}
  
  # Health checking
  health_check_interval: 30   # Health check interval (seconds)
  connection_pool_class: null # Custom connection pool class
```

### Redis Cluster Configuration

```yaml
redis:
  # Cluster mode settings
  cluster_mode: true
  startup_nodes:
    - host: redis-node-1.example.com
      port: 7001
    - host: redis-node-2.example.com  
      port: 7002
    - host: redis-node-3.example.com
      port: 7003
  
  # Cluster-specific options
  skip_full_coverage_check: false
  max_connections_per_node: 100
  readonly_mode: false
```

### Redis Sentinel Configuration

```yaml
redis:
  # Sentinel mode settings
  sentinel_mode: true
  sentinels:
    - host: sentinel-1.example.com
      port: 26379
    - host: sentinel-2.example.com
      port: 26379
    - host: sentinel-3.example.com
      port: 26379
  
  # Sentinel options
  service_name: mymaster
  sentinel_kwargs:
    password: sentinel_password
```

## Index Configuration

### Vector Index Settings

```yaml
index:
  # Basic index parameters
  algorithm: HNSW             # Index algorithm: HNSW, FLAT
  dimensions: 1536            # Vector dimensions
  distance_metric: L2         # Distance metric: L2, IP, COSINE
  initial_cap: 10000          # Initial index capacity
  block_size: 1024           # Memory block size
  
  # HNSW-specific parameters
  m: 16                      # Number of bi-directional links for every vertex
  ef_construction: 356       # Size of dynamic candidate list
  ef_runtime: 200           # Size of dynamic candidate list for search
  max_m: 16                 # Maximum number of connections for level 0
  max_m0: 32                # Maximum number of connections for other levels
  ml: 1.0                   # Level normalization factor
```

### Index Algorithm Details

#### HNSW (Hierarchical Navigable Small World)

```yaml
index:
  algorithm: HNSW
  m: 16                     # Typical range: 4-64
                           # Higher values: better recall, more memory
                           # Lower values: faster builds, less memory
  
  ef_construction: 356      # Typical range: 100-800  
                           # Higher values: better index quality, slower build
                           # Recommended: 2x ef_runtime
  
  ef_runtime: 200          # Typical range: 50-400
                           # Higher values: better recall, slower search
                           # Balance based on recall requirements
```

#### FLAT (Brute Force)

```yaml
index:
  algorithm: FLAT           # Exact search, 100% recall
  block_size: 1024         # Memory allocation block size
  initial_cap: 10000       # Initial vector capacity
```

### Distance Metrics

```yaml
index:
  # L2 (Euclidean) distance
  distance_metric: L2       # Best for: general purpose, normalized vectors
  
  # Inner Product  
  distance_metric: IP       # Best for: similarity scoring, recommendation
  
  # Cosine similarity
  distance_metric: COSINE   # Best for: text embeddings, angular similarity
```

## Workload Configuration

### Concurrency Settings

```yaml
workload:
  # Thread and client configuration
  n_threads: 8              # Number of worker threads
  n_clients_per_thread: 125 # Clients per thread
  total_clients: 1000       # Alternative: specify total directly
  
  # Batch processing
  batch_size: 1000          # Operations per batch
  max_batch_size: 5000      # Maximum batch size limit
  min_batch_size: 10        # Minimum batch size limit
  adaptive_batching: false  # Enable adaptive batch sizing
```

### Operation Settings

```yaml
workload:
  # Timeout and retry configuration
  operation_timeout: 30.0   # Operation timeout (seconds)
  max_retries: 3           # Maximum retry attempts
  retry_delay: 1.0         # Base retry delay (seconds)
  exponential_backoff: true # Use exponential backoff
  jitter: true             # Add random jitter to retries
  
  # Rate limiting
  max_operations_per_second: null # Rate limit (null = unlimited)
  burst_capacity: 1000     # Burst operation capacity
```

### Memory Management

```yaml
workload:
  # Memory optimization
  vector_cache_size: 10000  # Number of vectors to cache
  enable_vector_pooling: true # Reuse vector objects
  gc_frequency: 1000       # Garbage collection frequency
  
  # Resource limits
  max_memory_usage_mb: 8192 # Maximum memory usage
  memory_warning_threshold: 0.8 # Warning at 80% usage
```

## Monitoring Configuration

### Metrics Collection

```yaml
monitoring:
  # Collection settings
  collection_enabled: true  # Enable metrics collection
  sampling_interval: 10.0  # Collection interval (seconds)
  buffer_size: 1000       # Metrics buffer size
  
  # Memory metrics
  memory_metrics:
    - rss_mb               # Resident Set Size
    - vms_mb               # Virtual Memory Size  
    - active_mb            # Active memory
    - resident_mb          # Resident memory
    - allocated_mb         # Allocated memory
    - fragmentation_ratio  # Memory fragmentation
    - peak_allocated_mb    # Peak allocated memory
    - cache_hit_ratio     # Cache hit ratio
```

### Performance Metrics

```yaml
monitoring:
  # Performance tracking
  performance_metrics:
    - operations_per_second # Throughput
    - latency_p50          # 50th percentile latency
    - latency_p95          # 95th percentile latency  
    - latency_p99          # 99th percentile latency
    - latency_max          # Maximum latency
    - error_rate           # Error percentage
    - connection_count     # Active connections
    - queue_depth          # Operation queue depth
```

### Export Configuration

```yaml
monitoring:
  # Export formats
  export_format: json       # json, csv, prometheus, influxdb
  export_interval: 60      # Export interval (seconds)
  
  # Output destinations
  output_file: metrics.json # Local file output
  prometheus_port: 8080    # Prometheus metrics port
  influxdb_url: null       # InfluxDB endpoint
  
  # Retention settings
  retention_days: 7        # Days to retain metrics
  max_file_size_mb: 100   # Maximum output file size
```

## Environment Variables

Override configuration values using environment variables:

### Redis Settings
```bash
export VST_REDIS_HOST=redis.example.com
export VST_REDIS_PORT=6379
export VST_REDIS_PASSWORD=secret123
export VST_REDIS_DB=0
export VST_REDIS_SSL=true
export VST_REDIS_MAX_CONNECTIONS=2000
```

### Index Settings
```bash
export VST_INDEX_ALGORITHM=HNSW
export VST_INDEX_DIMENSIONS=1536
export VST_INDEX_M=16
export VST_INDEX_EF_CONSTRUCTION=356
export VST_INDEX_EF_RUNTIME=200
```

### Workload Settings
```bash
export VST_WORKLOAD_THREADS=16
export VST_WORKLOAD_CLIENTS=2000
export VST_WORKLOAD_BATCH_SIZE=1000
export VST_WORKLOAD_TIMEOUT=60
```

### Using Environment Variables in Config Files

```yaml
redis:
  host: "${VST_REDIS_HOST:-localhost}"
  port: "${VST_REDIS_PORT:-6379}"
  password: "${VST_REDIS_PASSWORD}"
  
workload:
  n_threads: "${VST_WORKLOAD_THREADS:-8}"
  batch_size: "${VST_WORKLOAD_BATCH_SIZE:-1000}"
```

## Configuration Validation

### Automatic Validation

The tool automatically validates configurations on startup:

```bash
# Validate configuration file
vst config validate --config my_config.yaml

# Check current configuration
vst config show

# Test Redis connection with current config
vst info redis
```

### Validation Rules

#### Redis Configuration
- Host must be reachable
- Port must be valid (1-65535)
- Database number must be 0-15
- SSL settings must be consistent

#### Index Configuration
- Algorithm must be HNSW or FLAT
- Dimensions must be positive integer
- Distance metric must be L2, IP, or COSINE
- HNSW parameters must be within valid ranges

#### Workload Configuration
- Thread count must be positive
- Client count must be positive
- Batch size must be between min and max limits
- Timeout values must be positive

### Custom Validation

Create custom validation rules:

```yaml
# config/validation.yaml
validation:
  redis:
    connection_timeout: 5.0
    required_modules:
      - RediSearch
      - RedisJSON
  
  index:
    max_dimensions: 4096
    min_ef_construction: 100
  
  workload:
    max_threads: 32
    max_clients: 10000
```

## Advanced Configuration

### Performance Tuning

#### High-Throughput Configuration

```yaml
# config/high_throughput.yaml
redis:
  max_connections: 5000
  socket_keepalive: true
  retry_on_timeout: false

workload:
  n_threads: 32
  n_clients_per_thread: 156
  batch_size: 2000
  operation_timeout: 10.0
  
index:
  algorithm: FLAT  # For maximum throughput
  block_size: 4096

monitoring:
  sampling_interval: 1.0
  buffer_size: 10000
```

#### Low-Latency Configuration

```yaml
# config/low_latency.yaml
redis:
  socket_timeout: 1.0
  socket_connect_timeout: 1.0
  max_connections: 100

workload:
  n_threads: 4
  n_clients_per_thread: 25
  batch_size: 1
  operation_timeout: 1.0

index:
  algorithm: HNSW
  ef_runtime: 50  # Lower for faster search

monitoring:
  sampling_interval: 0.1
```

#### Memory-Optimized Configuration

```yaml
# config/memory_optimized.yaml
redis:
  max_connections: 50

workload:
  n_threads: 2
  n_clients_per_thread: 25
  batch_size: 100
  vector_cache_size: 1000
  enable_vector_pooling: true

index:
  algorithm: HNSW
  m: 8  # Lower memory usage
  initial_cap: 1000
  block_size: 256

monitoring:
  buffer_size: 100
  memory_warning_threshold: 0.7
```

### Multi-Environment Configuration

#### Development Environment

```yaml
# config/environments/development.yaml
redis:
  host: localhost
  port: 6379
  db: 1

workload:
  n_threads: 2
  n_clients_per_thread: 10
  batch_size: 100

monitoring:
  sampling_interval: 1.0
  collection_enabled: true
  export_format: json
```

#### Staging Environment

```yaml
# config/environments/staging.yaml
redis:
  host: staging-redis.internal
  port: 6379
  password: "${STAGING_REDIS_PASSWORD}"
  ssl: true

workload:
  n_threads: 8
  n_clients_per_thread: 50
  batch_size: 500

monitoring:
  sampling_interval: 5.0
  export_format: prometheus
  prometheus_port: 8080
```

#### Production Environment

```yaml
# config/environments/production.yaml
redis:
  cluster_mode: true
  startup_nodes:
    - host: redis-1.prod.internal
      port: 7001
    - host: redis-2.prod.internal
      port: 7001
    - host: redis-3.prod.internal
      port: 7001
  password: "${PRODUCTION_REDIS_PASSWORD}"
  ssl: true
  max_connections: 2000

workload:
  n_threads: 16
  n_clients_per_thread: 125
  batch_size: 1000
  operation_timeout: 30.0

monitoring:
  sampling_interval: 10.0
  export_format: influxdb
  influxdb_url: "https://influxdb.prod.internal:8086"
  retention_days: 30
```

## Best Practices

### Configuration Management

1. **Use Environment-Specific Configs**
   ```bash
   # Use different configs per environment
   vst run --config config/environments/production.yaml
   ```

2. **Version Control Configurations**
   ```bash
   # Keep configs in version control
   git add config/
   git commit -m "Update production configuration"
   ```

3. **Secure Sensitive Values**
   ```yaml
   # Use environment variables for secrets
   redis:
     password: "${REDIS_PASSWORD}"
   ```

4. **Document Configuration Changes**
   ```yaml
   # Add comments explaining configuration choices
   workload:
     n_threads: 16  # Optimized for 16-core production servers
     batch_size: 1000  # Balance between throughput and memory
   ```

### Performance Optimization

1. **Start with Defaults**
   - Begin with default configuration
   - Profile and identify bottlenecks
   - Adjust specific parameters incrementally

2. **Match Hardware Resources**
   ```yaml
   workload:
     n_threads: 8  # Match CPU core count
     n_clients_per_thread: 125  # Adjust based on memory
   ```

3. **Monitor Resource Usage**
   ```yaml
   monitoring:
     memory_metrics: [rss_mb, fragmentation_ratio]
     performance_metrics: [operations_per_second, latency_p95]
   ```

4. **Test Configuration Changes**
   ```bash
   # Test new configuration with small dataset
   vst run quick --config new_config.yaml
   ```

### Troubleshooting Common Issues

#### Connection Issues

```yaml
# Increase timeouts for unreliable networks
redis:
  socket_timeout: 60.0
  socket_connect_timeout: 30.0
  retry_on_timeout: true
  max_retries: 5
```

#### Memory Issues

```yaml
# Reduce memory usage
workload:
  batch_size: 100  # Smaller batches
  vector_cache_size: 1000  # Smaller cache
  n_clients_per_thread: 50  # Fewer concurrent clients

monitoring:
  memory_warning_threshold: 0.6  # Earlier warnings
```

#### Performance Issues

```yaml
# Optimize for performance
redis:
  max_connections: 2000  # More connections
  socket_keepalive: true  # Reuse connections

workload:
  batch_size: 2000  # Larger batches
  operation_timeout: 60.0  # More time for operations

index:
  algorithm: FLAT  # Fastest search (if memory allows)
```

## Troubleshooting

### Configuration Validation Errors

**Error: Invalid algorithm 'HNWS'**
```yaml
# Fix typo in algorithm name
index:
  algorithm: HNSW  # Not 'HNWS'
```

**Error: ef_construction must be between 1 and 4096**
```yaml
# Adjust parameter to valid range
index:
  ef_construction: 356  # Not 5000
```

### Connection Errors

**Error: Connection refused**
```bash
# Check Redis is running
redis-cli ping

# Verify host and port
vst info redis --config your_config.yaml
```

**Error: Authentication failed**
```yaml
# Verify password configuration
redis:
  password: "${REDIS_PASSWORD}"
```

### Performance Issues

**Low throughput**
1. Increase batch size
2. Add more threads/clients
3. Check network latency
4. Verify Redis performance

**High memory usage**
1. Reduce batch size
2. Lower client count
3. Enable vector pooling
4. Use smaller cache size

**High latency**
1. Reduce ef_runtime for HNSW
2. Use FLAT algorithm
3. Increase operation timeout
4. Check system resources

### Getting Help

For additional help with configuration:

1. **Check built-in help**
   ```bash
   vst config --help
   vst run --help
   ```

2. **Validate configuration**
   ```bash
   vst config validate --config your_config.yaml
   ```

3. **Check system compatibility**
   ```bash
   vst info system
   vst info redis
   ```

4. **Enable debug logging**
   ```bash
   vst run --log-level DEBUG --config your_config.yaml
   ```
