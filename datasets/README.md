# Datasets Directory

This directory contains dataset files used for stress testing. **Dataset files are automatically ignored by git** to prevent large files from being committed to the repository.

## Current Dataset Files

The following files are present locally but ignored by git:

- `vector_database_wikipedia_articles_embedded.csv` - OpenAI Wikipedia embeddings (1.7GB, 763k vectors)
- `vector_database_wikipedia_articles_embedded.zip` - Compressed original dataset (699MB)
- `test_wiki_sample.vkv` - Sample VKV conversion (147MB, 25k vectors)
- `wiki_sample_1k.csv` - Small test sample (69MB, 1k vectors)

## Dataset Management

### Downloading Datasets
Use the provided scripts to download datasets:
```bash
# Download standard datasets
./setup_datasets.sh
```

### Preparing Datasets
Convert datasets to VKV format for stress testing:
```bash
# Analyze dataset
vst prep estimate datasets/vector_database_wikipedia_articles_embedded.csv

# Prepare for stress testing
vst prep prepare wikipedia_dataset datasets/vector_database_wikipedia_articles_embedded.csv
```

### Git Ignore Configuration

The following file types are automatically ignored:
- `*.csv`, `*.tsv` - Tabular data
- `*.npy`, `*.npz` - NumPy arrays
- `*.h5`, `*.hdf5` - HDF5 files
- `*.vkv` - Vector Key-Value format
- `*.zip`, `*.tar.gz` - Compressed files
- `*.bvecs`, `*.fvecs`, `*.ivecs` - Vector formats

## Storage Recommendations

- **Development**: Keep datasets locally in this directory
- **CI/CD**: Download datasets during test runs if needed
- **Production**: Store large datasets in S3 or cloud storage
- **Sharing**: Use the VST dataset preparation commands to create standardized formats

## File Sizes

Dataset files can be very large:
- Small test sets: 1MB - 100MB
- Medium datasets: 100MB - 1GB  
- Large datasets: 1GB - 10GB+
- Production datasets: 10GB+

Always verify you have sufficient disk space before downloading or generating datasets.
