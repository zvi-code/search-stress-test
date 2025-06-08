# High-Level Architecture 

## Goals:

Create a memory stress-testing package for Valkey-Search (redissearch compatible module)
Pre-fill with up to 1B vectors from public datasets (OpenAI, HuggingFace, BIGANN)
Test memory usage across different vector index lifecycle scenarios
Extend datasets by generating vectors with norms > max dataset norm, creating Expand_n(dataset) sets
Calculate recall using query datasets and ground-truth vectors
Support index shrinking via random vector deletion
Generate workloads alternating between extension/shrink operations with query workloads
Continuously collect memory metrics (INFO MEMORY + jemalloc stats)
Record results to CSV and Prometheus pushgateway
Support both single-instance and clustered Valkey deployments
Design for extensibility (future workload additions)

Constraints:

Must download and prepare public datasets for ingestion
Must support optional pre-fill skip
Vector extension must follow specific mathematical rules (||Vn||₂ > NormMax)
Key naming convention for extended vectors: expand_n_<original_key>
Query and ground-truth vectors must be naturally extended for Expand_n(dataset)


1. Module Breakdown
valkey-memory-stress/
├── core/
│   ├── dataset.py          # Dataset loading, vector operations, norm calculations
│   ├── vector_ops.py       # Vector generation, expansion logic (||Vn||₂ > NormMax)
│   ├── connection.py       # Connection pool management, async Redis operations
│   └── metrics.py          # Memory sampling, metric collection, CSV writing
│
├── workload/
│   ├── base.py            # Abstract workload class, plugin interface
│   ├── executor.py        # Multi-threaded workload orchestration
│   ├── ingest.py          # Parallel vector insertion workload
│   ├── query.py           # KNN query workload with recall calculation
│   └── shrink.py          # Random deletion workload
│
├── scenarios/
│   ├── loader.py          # YAML scenario parser
│   ├── runner.py          # Scenario execution engine
│   └── builtin/           # Pre-defined scenarios (grow-shrink patterns)
│
├── monitoring/
│   ├── collector.py       # Async INFO MEMORY collector
│   ├── aggregator.py      # Metric aggregation and percentile calculation
│   └── exporter.py        # CSV exporter (future: Prometheus)
│
└── cli/
    ├── main.py            # Typer CLI entry point
    └── config.py          # Configuration management
2. Data Flow Diagram
┌─────────────────┐     ┌──────────────────┐
│ OpenAI Dataset  │────▶│ Dataset Loader   │
│   (5M vectors)  │     │ (dataset.py)     │
└─────────────────┘     └────────┬─────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │ Vector Expander │
                        │ (vector_ops.py) │
                        └────────┬────────┘
                                 │
           ┌─────────────────────┴─────────────────────┐
           │                                           │
           ▼                                           ▼
    ┌──────────────┐                           ┌──────────────┐
    │ Ground Truth │                           │ Query Vectors│
    │   Storage    │                           │   Storage    │
    └──────┬───────┘                           └──────┬───────┘
           │                                           │
           └───────────────┬───────────────────────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │ Scenario Runner  │◀────┐
                  │  (runner.py)     │     │
                  └────────┬─────────┘     │
                           │               │
        ┌──────────────────┼───────────────┤
        │                  │               │
        ▼                  ▼               ▼
┌───────────────┐  ┌───────────────┐  ┌──────────────┐
│Ingest Workload│  │Query Workload │  │Shrink Workload│
│ (ingest.py)   │  │ (query.py)    │  │ (shrink.py)  │
└───────┬───────┘  └───────┬───────┘  └──────┬───────┘
        │                  │                  │
        └──────────────────┴──────────────────┘
                           │
                           ▼
                 ┌──────────────────┐      ┌────────────────┐
                 │ Connection Pool  │◀─────│ Async Monitor  │
                 │ (connection.py)  │      │ (collector.py) │
                 └────────┬─────────┘      └────────┬───────┘
                          │                         │
                          ▼                         │
                    ┌───────────┐                   │
                    │  Valkey   │◀──────────────────┘
                    │  Instance │     INFO MEMORY
                    └───────────┘
                          │
                          ▼
                  ┌───────────────┐
                  │ CSV Exporter  │
                  │ (exporter.py) │
                  └───────────────┘
3. Third-Party Libraries
yamldependencies:
  - numpy>=1.24.0          # Vector operations, norm calculations
  - redis>=5.0.0           # Async Redis client with connection pooling
  - psutil>=5.9.0          # System memory monitoring
  - prometheus-client>=0.19.0  # Future Prometheus support
  - typer>=0.9.0           # CLI framework
  - pyyaml>=6.0            # YAML configuration parsing
  - h5py>=3.10.0           # HDF5 dataset loading (OpenAI format)
  - aiofiles>=23.0         # Async file operations
  - pandas>=2.0.0          # CSV operations and data analysis
4. Workload Plugin Architecture
New workloads extend the base class:
python# workload/base.py
class BaseWorkload(ABC):
    @abstractmethod
    async def execute(self, 
                     connection_pool: ConnectionPool,
                     dataset: Dataset,
                     config: Dict[str, Any]) -> WorkloadResult:
        pass
    
    @abstractmethod
    def get_metrics(self) -> Dict[str, float]:
        pass

# Custom workload example structure:
# workload/custom_mixed.py
class MixedWorkload(BaseWorkload):
    async def execute(self, ...):
        # 70% queries, 30% updates implementation
        pass
Workloads register via decorator:
python@register_workload("mixed_70_30")
class MixedWorkload(BaseWorkload):
    ...
5. Key Assumptions

Memory Model: Valkey RSS/Active/Resident memory metrics accurately reflect vector index memory usage
Vector Distribution: OpenAI dataset vectors follow a normal distribution for norm calculations
Async Performance: Redis-py async operations with 1000 clients won't overwhelm Valkey's event loop
Recall Calculation: Ground truth for expanded vectors (Vn + original) maintains the same nearest neighbors plus the expansion offset
Thread Safety: Valkey-Search handles concurrent operations without data corruption
Batch Sizing: 1000 vectors per batch for insertion balances memory and performance
Key Naming: No existing keys match pattern expand_* in the target database
Memory Sampling: 10-second intervals provide sufficient granularity without impacting performance
CSV Atomicity: CSV writes are line-buffered to prevent data loss on interruption
Dataset Format: OpenAI dataset is in HDF5 format with 'train', 'test', and 'neighbors' groups

Please confirm these assumptions or provide corrections before I proceed with the detailed design.