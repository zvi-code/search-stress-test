# CLI Reference Guide

Quick reference for all Valkey Stress Test CLI commands.

## Command Structure

```
vst [GLOBAL_OPTIONS] COMMAND [COMMAND_OPTIONS] [ARGUMENTS]
```

## Global Options

| Option | Short | Description |
|--------|-------|-------------|
| `--verbose` | `-v` | Enable verbose output |
| `--quiet` | `-q` | Suppress non-error output |
| `--help` | | Show help message |

## Commands Overview

| Command | Description |
|---------|-------------|
| `run` | Execute stress test scenarios |
| `dataset` | Manage datasets |
| `info` | Display system information |
| `validate` | Validate configurations |
| `version` | Show version information |

## Run Commands

### `vst run quick`

Quick stress test with default settings.

```bash
vst run quick [OPTIONS]
```

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--dataset` | `-d` | TEXT | openai-5m | Dataset to use |
| `--workload` | `-w` | TEXT | mixed | Workload type |
| `--duration` | `-t` | INTEGER | 300 | Duration in seconds |
| `--output` | `-o` | PATH | None | Output directory |

**Examples:**
```bash
# 10-minute test
vst run quick --duration 600

# With specific dataset
vst run quick --dataset sift-1m

# Custom output location
vst run quick --output ./test-results
```

### `vst run scenario`

Run a detailed stress test scenario.

```bash
vst run scenario [OPTIONS] SCENARIO_FILE
```

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--config` | `-c` | PATH | None | Configuration file |
| `--output` | `-o` | PATH | None | Output directory |
| `--dry-run` | | FLAG | False | Validate without executing |

**Examples:**
```bash
# Built-in scenario
vst run scenario continuous_growth

# Custom scenario
vst run scenario ./my-scenario.yaml

# With custom config
vst run scenario ./scenario.yaml --config ./config.yaml

# Validation only
vst run scenario ./scenario.yaml --dry-run
```

### `vst run list-scenarios`

List available built-in scenarios.

```bash
vst run list-scenarios
```

### `vst run validate`

Validate a scenario file.

```bash
vst run validate SCENARIO_FILE
```

## Dataset Commands

### `vst dataset list`

List available datasets.

```bash
vst dataset list
```

### `vst dataset download`

Download a dataset.

```bash
vst dataset download [OPTIONS] DATASET_NAME
```

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--output` | `-o` | PATH | ./datasets | Output directory |

**Examples:**
```bash
vst dataset download openai-5m
vst dataset download sift-1m --output /data
```

### `vst dataset info`

Display dataset information.

```bash
vst dataset info DATASET_PATH
```

**Example:**
```bash
vst dataset info ./datasets/openai-5m.h5
```

### `vst dataset prepare`

Prepare a dataset for use.

```bash
vst dataset prepare [OPTIONS] INPUT_PATH OUTPUT_PATH
```

## Info Commands

### `vst info system`

Display system information.

```bash
vst info system
```

### `vst info redis`

Display Redis/Valkey server information.

```bash
vst info redis [OPTIONS]
```

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--host` | `-h` | TEXT | localhost | Redis host |
| `--port` | `-p` | INTEGER | 6379 | Redis port |
| `--password` | `-a` | TEXT | None | Redis password |

**Examples:**
```bash
vst info redis
vst info redis --host myredis.example.com --port 6380
vst info redis --password mypassword
```

### `vst info workloads`

List available workloads.

```bash
vst info workloads
```

## Validate Commands

### `vst validate scenario`

Validate a scenario file.

```bash
vst validate scenario SCENARIO_FILE
```

### `vst validate config`

Validate a configuration file.

```bash
vst validate config CONFIG_FILE
```

## Version Command

### `vst version`

Display version information.

```bash
vst version
```

## Exit Codes

| Code | Description |
|------|-------------|
| 0 | Success |
| 1 | General error |
| 2 | Invalid command line arguments |

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `VST_CONFIG_FILE` | Default configuration file | None |
| `VST_OUTPUT_DIR` | Default output directory | ./output |
| `VST_REDIS_HOST` | Default Redis host | localhost |
| `VST_REDIS_PORT` | Default Redis port | 6379 |

## Common Workflows

### Basic Testing Workflow

```bash
# 1. Check system
vst info system

# 2. Check Redis connection
vst info redis

# 3. Download dataset
vst dataset download openai-5m

# 4. Run quick test
vst run quick --duration 300

# 5. Check results
ls ./output/
```

### Custom Scenario Workflow

```bash
# 1. Validate scenario
vst validate scenario ./my-scenario.yaml

# 2. Dry run
vst run scenario ./my-scenario.yaml --dry-run

# 3. Execute with custom output
vst run scenario ./my-scenario.yaml --output ./custom-test

# 4. Check results
vst dataset info ./custom-test/metrics.csv
```

### Dataset Management Workflow

```bash
# 1. List available datasets
vst dataset list

# 2. Download multiple datasets
vst dataset download openai-5m --output ./datasets
vst dataset download sift-1m --output ./datasets

# 3. Check dataset information
vst dataset info ./datasets/openai-5m.h5
vst dataset info ./datasets/sift-1m.h5

# 4. Prepare custom dataset (if needed)
vst dataset prepare ./my-vectors.npy ./datasets/custom.h5
```

## Tips and Best Practices

### Performance Optimization

1. **Connection Tuning**: Use appropriate `max_connections` in config
2. **Batch Sizing**: Adjust `batch_size` based on available memory
3. **Thread Configuration**: Set `n_threads` to match CPU cores
4. **Client Scaling**: Balance `n_clients` with system resources

### Error Handling

1. **Always validate** scenarios before execution
2. **Use dry-run** for complex scenarios
3. **Check system resources** before large tests
4. **Monitor output directory** space

### Debugging

1. **Use verbose mode** (`--verbose`) for detailed logs
2. **Start with small tests** to verify setup
3. **Check Redis connectivity** with `vst info redis`
4. **Validate all configurations** before running

### Resource Management

1. **Monitor memory usage** during tests
2. **Clean up output directories** after tests
3. **Use appropriate dataset sizes** for your system
4. **Set reasonable timeouts** in configurations
