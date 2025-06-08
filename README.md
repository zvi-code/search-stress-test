# Valkey Stress Test

Memory stress testing tool for Valkey-Search with vector operations.

## Installation

```bash
poetry install
```

## Quick Start

```bash
# Run a scenario
vst run scenario config/scenarios/grow_shrink_grow.yaml

# Download dataset
vst dataset download openai-5m

# Validate configuration
vst validate scenario my_scenario.yaml
```

## Development

See `docs/` for detailed documentation.
