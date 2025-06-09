# Valkey Memory Stress Testing Tool

A comprehensive memory stress testing tool for Valkey-Search with vector operations. This tool helps you test memory usage patterns, performance characteristics, and stability of Valkey instances under various vector workloads.

## Features

- 🚀 **Multiple Workload Types**: Built-in support for ingestion, querying, and data shrinking workloads
- 📊 **Memory Monitoring**: Real-time memory usage tracking with detailed metrics
- 🎯 **Scenario-Based Testing**: Define complex test scenarios with multiple phases
- 📈 **Performance Metrics**: Comprehensive performance and latency measurements
- 🔧 **Configurable**: Flexible configuration for different testing needs
- 📦 **Dataset Management**: Built-in dataset download and management
- 🛡️ **Validation**: Configuration and scenario validation before execution

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [CLI Reference](#cli-reference)
- [Configuration](#configuration)
- [Scenarios](#scenarios)
- [Datasets](#datasets)
- [Examples](#examples)
- [Development](#development)
- [Quick Install Guide](QUICK_INSTALL.md)
- [Detailed Installation Guide](INSTALL.md)

## Installation

> **Note**: This package is currently in development and not yet published to PyPI. Install from source using the instructions below.

### System Requirements Check

**Before installing**, run our system check to ensure compatibility:

```bash
# 1. Clone the repository
git clone https://github.com/your-org/valkey_stress_test.git
cd valkey_stress_test

# 2. Check your system requirements
python3 setup_check.py
```

This will check your Python version (3.10+ required) and provide specific installation guidance for your system.

### Quick Installation

```bash
# After confirming system compatibility:

# 1. Clone the repository (if not already done)
git clone https://github.com/your-org/valkey_stress_test.git
cd valkey_stress_test

# 2. Install dependencies and package
pip install -r requirements.txt
pip install -e .

# 3. Verify installation
python verify_installation.py
```

**For AWS EC2 (Amazon Linux):**
```bash
git clone https://github.com/your-org/valkey_stress_test.git
cd valkey_stress_test
./setup_ec2.sh  # Automated setup for Amazon Linux
```

### Prerequisites

- **Python 3.10 or higher** - Check with `python3 --version`
- **Git** - For cloning the repository  
- **Redis/Valkey instance** - With Search module enabled

> **⚠️ Python Version Issue?** If you have Python < 3.10, see our [detailed installation guide](INSTALL.md) for upgrade solutions including pyenv, conda, and Docker options.

### Troubleshooting Common Issues

**Problem: `vst` command not found**
```bash
# Solution: Ensure package is installed in editable mode
pip install -e .
```

**Problem: Python version < 3.10**
```bash
# Quick check
python3 setup_check.py

# Solutions provided based on your system
```

**Problem: Installation fails on AWS EC2/Cloud servers**
- See [AWS EC2 Installation Guide](INSTALL.md#aws-ec2-installation) in our detailed installation guide

### Installation Options

**Option 1: Using pip (Recommended for users)**
```bash
git clone https://github.com/your-org/valkey_stress_test.git
cd valkey_stress_test
pip install -r requirements.txt
pip install -e .
```

**Option 2: Using Poetry (For developers)**
```bash
git clone https://github.com/your-org/valkey_stress_test.git
cd valkey_stress_test
poetry install
poetry shell
```

**Option 3: With virtual environment (Recommended)**
```bash
git clone https://github.com/your-org/valkey_stress_test.git
cd valkey_stress_test
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

### Detailed Installation Guide

For complete installation instructions including troubleshooting, see **[INSTALL.md](INSTALL.md)**.

## Quick Start

### 1. Start Valkey/Redis with Search Module

```bash
# Using Docker (recommended)
docker run -d --name valkey-search \
  -p 6379:6379 \
  redis/redis-stack-server:latest

# Or using Valkey
docker run -d --name valkey \
  -p 6379:6379 \
  valkey/valkey:latest
```

### 2. Run a Quick Test

```bash
# Run a quick 5-minute stress test
vst run quick --duration 300

# Run with custom dataset
vst run quick --dataset openai-5m --duration 600
```

### 3. Check System Information

```bash
# Check system information
vst info system

# Check Redis/Valkey connection
vst info redis

# List available workloads
vst info workloads
```

### 4. Download and Use Datasets

```bash
# List available datasets
vst dataset list

# Download a dataset
vst dataset download openai-5m --output ./datasets

# Get dataset information
vst dataset info ./datasets/openai-5m.h5
```

### 5. Run Tests

```bash
# Run unit tests (no external dependencies)
python run_tests.py

# Run with coverage report
python run_tests.py --coverage

# Quick shell script version
./run_tests.sh
```

## CLI Reference

### Main Commands

```bash
vst [OPTIONS] COMMAND [ARGS]...
```

**Global Options:**
- `--verbose, -v`: Enable verbose output
- `--quiet, -q`: Suppress non-error output
- `--help`: Show help message

### Run Commands

#### Quick Test
```bash
vst run quick [OPTIONS]
```

Run a quick stress test with default settings.

**Options:**
- `--dataset, -d TEXT`: Dataset to use (default: openai-5m)
- `--workload, -w TEXT`: Workload type (default: mixed)
- `--duration, -t INTEGER`: Duration in seconds (default: 300)
- `--output, -o PATH`: Output directory

**Examples:**
```bash
# Quick 10-minute test
vst run quick --duration 600

# Quick test with specific dataset
vst run quick --dataset sift-1m --duration 300

# Quick test with output directory
vst run quick --output ./test-results
```

#### Scenario Test
```bash
vst run scenario [OPTIONS] SCENARIO_FILE
```

Run a detailed stress test scenario.

**Options:**
- `--config, -c PATH`: Configuration file
- `--output, -o PATH`: Output directory
- `--dry-run`: Validate without executing

**Examples:**
```bash
# Run built-in scenario
vst run scenario continuous_growth

# Run custom scenario
vst run scenario ./my-scenario.yaml --output ./results

# Validate scenario without running
vst run scenario ./my-scenario.yaml --dry-run
```

#### List Scenarios
```bash
vst run list-scenarios
```

List all available built-in scenarios.

#### Validate Scenario
```bash
vst run validate SCENARIO_FILE
```

Validate a scenario file without executing it.

### Dataset Commands

#### List Datasets
```bash
vst dataset list
```

List all available datasets for download.

#### Download Dataset
```bash
vst dataset download [OPTIONS] DATASET_NAME
```

Download a dataset for testing.

**Options:**
- `--output, -o PATH`: Output directory (default: ./datasets)

**Examples:**
```bash
# Download to default location
vst dataset download openai-5m

# Download to specific directory
vst dataset download sift-1m --output /data/datasets
```

#### Dataset Information
```bash
vst dataset info DATASET_PATH
```

Display information about a dataset file.

**Example:**
```bash
vst dataset info ./datasets/openai-5m.h5
```

#### Prepare Dataset
```bash
vst dataset prepare [OPTIONS] INPUT_PATH OUTPUT_PATH
```

Prepare a dataset for use (convert format, sample, etc.).

### Information Commands

#### System Information
```bash
vst info system
```

Display system information including CPU, memory, and Python version.

#### Redis/Valkey Information
```bash
vst info redis [OPTIONS]
```

Display Redis/Valkey server information.

**Options:**
- `--host, -h TEXT`: Redis host (default: localhost)
- `--port, -p INTEGER`: Redis port (default: 6379)
- `--password, -a TEXT`: Redis password

**Example:**
```bash
vst info redis --host myredis.example.com --port 6380
```

#### Workload Information
```bash
vst info workloads
```

List all available workload types.

### Validation Commands

#### Validate Scenario
```bash
vst validate scenario SCENARIO_FILE
```

Validate a scenario configuration file.

#### Validate Configuration
```bash
vst validate config CONFIG_FILE
```

Validate a configuration file.

**Examples:**
```bash
vst validate scenario ./scenarios/my-test.yaml
vst validate config ./config/production.yaml
```

### Version Information
```bash
vst version
```

Display version information.

## Configuration

### Configuration File Structure

Configuration files use YAML format:

```yaml
# config.yaml
redis:
  host: localhost
  port: 6379
  db: 0
  max_connections: 1000
  
index:
  algorithm: HNSW
  dimensions: 1536
  distance_metric: L2
  m: 16
  ef_construction: 356
  ef_runtime: 200
  
workload:
  n_threads: 8
  n_clients_per_thread: 125
  batch_size: 1000
  operation_timeout: 30.0
  
monitoring:
  sampling_interval: 10.0
  memory_metrics:
    - rss_mb
    - active_mb
    - resident_mb
    - allocated_mb
    - fragmentation_ratio
    
output:
  csv_path: output/metrics.csv
  summary_path: output/summary.csv
```

### Configuration Sections

#### Redis Connection
- `host`: Redis/Valkey server hostname
- `port`: Server port number
- `db`: Database number
- `max_connections`: Maximum connection pool size

#### Vector Index
- `algorithm`: Vector index algorithm (HNSW, FLAT)
- `dimensions`: Vector dimensions
- `distance_metric`: Distance metric (L2, IP, COSINE)
- `m`: HNSW parameter M
- `ef_construction`: HNSW construction parameter
- `ef_runtime`: HNSW runtime parameter

#### Workload
- `n_threads`: Number of worker threads
- `n_clients_per_thread`: Clients per thread
- `batch_size`: Operations per batch
- `operation_timeout`: Timeout for operations

## Scenarios

### Scenario File Structure

Scenarios define complex test workflows:

```yaml
# scenario.yaml
name: my_test_scenario
description: Custom memory stress test

dataset: openai-5m

global_config:
  n_threads: 4
  n_clients: 500
  batch_size: 1000

steps:
  - name: initial_load
    type: workload
    workload: ingest
    parameters:
      target_vectors: 1000000
      
  - name: query_phase
    type: workload
    workload: query
    duration_seconds: 300
    parameters:
      queries_per_second: 100
      k: 10
      
  - name: checkpoint
    type: checkpoint
    parameters:
      collect_full_metrics: true
      
  - name: cleanup
    type: workload
    workload: shrink
    parameters:
      target_vectors: 500000
```

### Step Types

#### Workload Steps
Execute a specific workload type:
```yaml
- name: load_data
  type: workload
  workload: ingest
  parameters:
    target_vectors: 1000000
```

#### Wait Steps
Pause execution for a specified condition:
```yaml
- name: stabilize
  type: wait
  wait_condition:
    type: duration
    seconds: 30
```

#### Checkpoint Steps
Collect metrics and create measurement points:
```yaml
- name: measure
  type: checkpoint
  parameters:
    collect_full_metrics: true
    export_data: true
```

### Built-in Scenarios

- `continuous_growth`: Tests memory patterns with continuous data growth
- `grow_shrink_grow`: Tests memory patterns with growth and shrinkage cycles

View scenarios:
```bash
vst run list-scenarios
```

## Datasets

### Available Datasets

- `openai-5m`: OpenAI embeddings dataset (5M vectors, 1536 dimensions)
- `sift-1m`: SIFT feature vectors (1M vectors, 128 dimensions)
- `gist-1m`: GIST descriptors (1M vectors, 960 dimensions)

### Dataset Operations

```bash
# List available datasets
vst dataset list

# Download a dataset
vst dataset download openai-5m

# Get dataset information
vst dataset info ./datasets/openai-5m.h5

# Prepare custom dataset
vst dataset prepare ./raw-data.npy ./processed-data.h5
```

## Examples

### Example 1: Quick Performance Test

```bash
# Run a 10-minute performance test
vst run quick --duration 600 --output ./performance-test

# Check the results
ls ./performance-test/
```

### Example 2: Memory Growth Test

Create a scenario file `memory-growth.yaml`:

```yaml
name: memory_growth_test
description: Test memory usage during data ingestion

dataset: openai-5m

global_config:
  n_threads: 4
  n_clients: 200
  batch_size: 500

steps:
  - name: baseline
    type: checkpoint
    parameters:
      collect_full_metrics: true
      
  - name: load_1m
    type: workload
    workload: ingest
    parameters:
      target_vectors: 1000000
      
  - name: checkpoint_1m
    type: checkpoint
    parameters:
      collect_full_metrics: true
      
  - name: load_2m
    type: workload
    workload: ingest
    parameters:
      target_vectors: 2000000
      
  - name: final_checkpoint
    type: checkpoint
    parameters:
      collect_full_metrics: true
```

Run the test:

```bash
vst run scenario memory-growth.yaml --output ./memory-test
```

### Example 3: Mixed Workload Test

Create `mixed-workload.yaml`:

```yaml
name: mixed_workload
description: Test with mixed read/write operations

dataset: sift-1m

global_config:
  n_threads: 8
  n_clients: 400
  batch_size: 200

steps:
  - name: initial_load
    type: workload
    workload: ingest
    parameters:
      target_vectors: 500000
      
  - name: mixed_operations
    type: workload
    workload: query
    duration_seconds: 600
    parameters:
      queries_per_second: 50
      k: 20
```

Run with custom configuration:

```bash
vst run scenario mixed-workload.yaml \
  --config ./config/high-performance.yaml \
  --output ./mixed-test
```

### Example 4: Validation and Debugging

```bash
# Validate scenario before running
vst run validate ./my-scenario.yaml

# Check Redis connection
vst info redis --host production-redis.example.com

# Run with verbose output
vst --verbose run scenario ./my-scenario.yaml

# Dry run to check configuration
vst run scenario ./my-scenario.yaml --dry-run
```

## Development

### Setting Up Development Environment

```bash
# Clone repository
git clone <repository-url>
cd valkey_stress_test

# Install development dependencies
poetry install --with dev

# Activate virtual environment
poetry shell

# Run tests
pytest

# Run with coverage
pytest --cov=valkey_stress_test

# Format code
black .
ruff check .
```

### Running Tests

We provide a comprehensive test runner that handles all setup automatically:

```bash
# Run all unit tests (default, no Redis required)
python run_tests.py

# Run with coverage report
python run_tests.py --coverage

# Run integration tests (requires Redis/Valkey)
python run_tests.py --type integration

# Quick tests only (faster)
python run_tests.py --quick

# Verbose output with fail-fast
python run_tests.py --verbose --failfast

# Shell script alternative
./run_tests.sh --coverage
```

See [TESTING.md](TESTING.md) for complete testing documentation.

#### Direct pytest usage (advanced)
```bash
# If you prefer using pytest directly
pytest tests/unit/     # Unit tests only
pytest tests/integration/  # Integration tests
pytest -v              # Verbose output
```

### Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Run the test suite
6. Submit a pull request

## Troubleshooting

### Common Issues

#### Connection Refused
```
Error: Redis connection refused
```
**Solution**: Ensure Redis/Valkey is running and accessible:
```bash
# Check if Redis is running
redis-cli ping

# Start Redis with Docker
docker run -d -p 6379:6379 redis/redis-stack-server:latest
```

#### Permission Denied
```
Error: Permission denied writing to output directory
```
**Solution**: Check directory permissions:
```bash
# Create output directory with proper permissions
mkdir -p ./output
chmod 755 ./output
```

#### Memory Issues
```
Error: Out of memory during test execution
```
**Solution**: Reduce test parameters:
- Decrease `n_clients` in configuration
- Reduce `batch_size`
- Use smaller `target_vectors`

#### Module Not Found
```
Error: Redis Search module not found
```
**Solution**: Use Redis Stack or enable Search module:
```bash
# Use Redis Stack (includes Search module)
docker run -d -p 6379:6379 redis/redis-stack-server:latest
```

### Debug Mode

Enable verbose logging for debugging:

```bash
# Verbose output
vst --verbose run scenario ./my-scenario.yaml

# Check system information
vst info system

# Validate configuration
vst validate scenario ./my-scenario.yaml
vst validate config ./my-config.yaml
```

### Performance Tips

1. **Optimize Connection Pool**: Adjust `max_connections` based on your Redis setup
2. **Tune Batch Size**: Larger batches can improve throughput but increase memory usage
3. **Monitor System Resources**: Use `vst info system` to check available resources
4. **Use Appropriate Dataset Size**: Start with smaller datasets for testing

## License

[Add your license information here]

## Support

- Documentation: See `docs/` directory
- Issues: [GitHub Issues](link-to-issues)
- Discussions: [GitHub Discussions](link-to-discussions)
