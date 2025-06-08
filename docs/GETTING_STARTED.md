# Getting Started Guide

This guide will help you get up and running with the Valkey Memory Stress Testing Tool in just a few minutes.

## Prerequisites Checklist

Before you begin, ensure you have:

- [ ] Python 3.10 or higher installed
- [ ] A running Valkey/Redis instance with Search module
- [ ] At least 4GB of available RAM
- [ ] Network access for dataset downloads (optional)

## Step 1: Installation

### Option A: Development Installation (Recommended)

```bash
# Clone the repository
git clone <repository-url>
cd valkey_stress_test

# Install with Poetry
poetry install

# Activate the environment
poetry shell
```

### Option B: Direct Installation

```bash
# Install from source
pip install -e .
```

## Step 2: Start Redis/Valkey

### Using Docker (Easiest)

```bash
# Start Redis Stack (includes Search module)
docker run -d --name redis-stack \
  -p 6379:6379 \
  redis/redis-stack-server:latest

# Verify it's running
docker ps
```

### Using Local Installation

```bash
# Start Redis with Search module
redis-server --loadmodule /path/to/redisearch.so

# Or start Valkey
valkey-server --loadmodule /path/to/redisearch.so
```

## Step 3: Verify Installation

```bash
# Check the tool is working
vst version

# Check system information
vst info system

# Test Redis connection
vst info redis
```

Expected output for `vst info redis`:
```
Redis/Valkey Server Information (localhost:6379)
==================================================

Server:
  redis_version: 7.2.0
  redis_mode: standalone
  os: Linux
...
```

## Step 4: Run Your First Test

### Quick 5-Minute Test

```bash
# Run a quick stress test
vst run quick --duration 300

# This will:
# 1. Use the default openai-5m dataset
# 2. Run a mixed workload for 5 minutes
# 3. Save results to ./output/
```

### Check the Results

```bash
# List output files
ls -la ./output/

# You should see files like:
# - metrics.csv        (detailed metrics)
# - summary.csv        (test summary)
# - scenario_log.json  (execution log)
```

## Step 5: Understanding the Output

### Metrics File (`metrics.csv`)
Contains time-series data of memory usage:
- `timestamp`: When the measurement was taken
- `rss_mb`: Resident Set Size in MB
- `active_mb`: Active memory in MB
- `allocated_mb`: Allocated memory in MB
- `fragmentation_ratio`: Memory fragmentation ratio

### Summary File (`summary.csv`)
Contains aggregated test results:
- Total vectors processed
- Average throughput
- Memory usage statistics
- Performance metrics

## Step 6: Try a Custom Scenario

### Create a Simple Scenario

Create a file called `my-first-scenario.yaml`:

```yaml
name: my_first_test
description: My first custom stress test

dataset: openai-5m

global_config:
  n_threads: 2
  n_clients: 100
  batch_size: 500

steps:
  # Load some data
  - name: load_data
    type: workload
    workload: ingest
    parameters:
      target_vectors: 100000
      
  # Take a measurement
  - name: checkpoint_after_load
    type: checkpoint
    parameters:
      collect_full_metrics: true
      
  # Run some queries
  - name: query_test
    type: workload
    workload: query
    duration_seconds: 60
    parameters:
      queries_per_second: 10
      k: 10
```

### Validate and Run the Scenario

```bash
# First, validate the scenario
vst validate scenario my-first-scenario.yaml

# Run a dry-run to check everything
vst run scenario my-first-scenario.yaml --dry-run

# Actually run the scenario
vst run scenario my-first-scenario.yaml --output ./my-test
```

## Step 7: Dataset Management

### Download a Dataset

```bash
# List available datasets
vst dataset list

# Download a smaller dataset for testing
vst dataset download sift-1m --output ./datasets

# Check dataset information
vst dataset info ./datasets/sift-1m.h5
```

## Common Next Steps

### Performance Tuning

1. **Adjust thread count** based on your CPU:
   ```bash
   # Check CPU count
   vst info system
   
   # Modify scenario to use appropriate thread count
   ```

2. **Scale client connections** based on your system:
   ```yaml
   global_config:
     n_threads: 4        # Number of CPU cores
     n_clients: 200      # Start with 50 per thread
     batch_size: 1000    # Increase for better throughput
   ```

### Memory Testing

1. **Start small** and increase load:
   ```yaml
   steps:
     - name: small_load
       type: workload
       workload: ingest
       parameters:
         target_vectors: 10000
         
     - name: medium_load
       type: workload
       workload: ingest
       parameters:
         target_vectors: 100000
         
     - name: large_load
       type: workload
       workload: ingest
       parameters:
         target_vectors: 1000000
   ```

2. **Add checkpoints** to measure memory at each stage:
   ```yaml
   - name: checkpoint_after_small
     type: checkpoint
     parameters:
       collect_full_metrics: true
   ```

### Configuration Customization

Create a custom configuration file `my-config.yaml`:

```yaml
redis:
  host: localhost
  port: 6379
  max_connections: 500  # Adjust based on your Redis setup

workload:
  n_threads: 4
  n_clients_per_thread: 50
  batch_size: 500
  operation_timeout: 30.0

monitoring:
  sampling_interval: 5.0  # More frequent sampling
```

Use it with scenarios:
```bash
vst run scenario my-scenario.yaml --config my-config.yaml
```

## Troubleshooting

### Connection Issues

```bash
# Problem: "Connection refused"
# Solution: Check if Redis is running
docker ps
redis-cli ping

# Problem: "Search module not found"
# Solution: Use Redis Stack
docker run -d -p 6379:6379 redis/redis-stack-server:latest
```

### Memory Issues

```bash
# Problem: Out of memory
# Solution: Reduce test parameters
# In scenario file, reduce:
# - target_vectors
# - n_clients
# - batch_size
```

### Performance Issues

```bash
# Problem: Tests are too slow
# Solution: Check system resources
vst info system

# And tune configuration:
# - Increase batch_size
# - Adjust n_threads to match CPU cores
# - Reduce sampling_interval
```

## Next Steps

1. **Read the full documentation**: Check `/docs` directory
2. **Explore built-in scenarios**: `vst run list-scenarios`
3. **Create complex scenarios**: Use multiple workload types
4. **Monitor production systems**: Use longer-running tests
5. **Contribute**: See development guide in README

## Getting Help

- Use `--help` with any command for detailed options
- Check the CLI Reference: `docs/CLI_REFERENCE.md`
- Enable verbose mode: `vst --verbose <command>`
- Validate configurations before running: `vst validate <type> <file>`

## Example: Complete Workflow

Here's a complete example workflow:

```bash
# 1. Setup
docker run -d -p 6379:6379 redis/redis-stack-server:latest
vst info redis

# 2. Quick test
vst run quick --duration 60

# 3. Custom scenario
cat > test-scenario.yaml << EOF
name: learning_test
description: Learning the tool
dataset: openai-5m
global_config:
  n_threads: 2
  n_clients: 50
  batch_size: 200
steps:
  - name: load_small
    type: workload
    workload: ingest
    parameters:
      target_vectors: 50000
  - name: measure
    type: checkpoint
    parameters:
      collect_full_metrics: true
  - name: query_test
    type: workload
    workload: query
    duration_seconds: 30
    parameters:
      queries_per_second: 5
EOF

# 4. Run custom scenario
vst validate scenario test-scenario.yaml
vst run scenario test-scenario.yaml --output ./learning-test

# 5. Check results
ls -la ./learning-test/
head ./learning-test/metrics.csv
```

You're now ready to start stress testing your Valkey/Redis instances! 🚀
