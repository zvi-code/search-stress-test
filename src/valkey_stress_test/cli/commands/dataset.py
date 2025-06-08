"""Dataset management commands."""

import typer
import h5py
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any
import urllib.request
import logging

from ...core.dataset import Dataset

app = typer.Typer()
logger = logging.getLogger(__name__)


# Known dataset URLs and metadata
DATASETS = {
    "openai-5m": {
        "url": "https://cdn.openai.com/API/examples/data/vector_database_wikipedia_articles_embedded.zip",
        "description": "OpenAI 5M vectors from Wikipedia articles",
        "format": "zip",
        "dimensions": 1536,
        "size_mb": 2000,
    },
    "sift-1m": {
        "url": "http://corpus-texmex.irisa.fr/sift.tar.gz",
        "description": "SIFT 1M benchmark dataset",
        "format": "tar.gz", 
        "dimensions": 128,
        "size_mb": 100,
    }
}


def _download_with_progress(url: str, output_path: Path) -> None:
    """Download a file with progress bar."""
    import urllib.request
    
    def progress_hook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = (downloaded / total_size) * 100
            typer.echo(f"\rDownloading: {percent:.1f}% ({downloaded / (1024*1024):.1f} MB)", nl=False)
    
    urllib.request.urlretrieve(url, output_path, reporthook=progress_hook)
    typer.echo()  # New line after progress


@app.command()
def download(
    name: str = typer.Argument(..., help="Dataset name (e.g., openai-5m, sift-1m)"),
    output_dir: Optional[Path] = typer.Option(None, "--output", "-o", help="Output directory"),
    force: bool = typer.Option(False, "--force", "-f", help="Overwrite existing files"),
):
    """Download a dataset for testing."""
    if name not in DATASETS:
        available = ", ".join(DATASETS.keys())
        typer.echo(f"❌ Unknown dataset: {name}")
        typer.echo(f"Available datasets: {available}")
        raise typer.Exit(1)
    
    dataset_info = DATASETS[name]
    
    # Setup output directory
    if output_dir is None:
        output_dir = Path("./datasets")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine output filename
    url = dataset_info["url"]
    filename = Path(url).name
    output_path = output_dir / filename
    
    # Check if file already exists
    if output_path.exists() and not force:
        typer.echo(f"❌ File already exists: {output_path}")
        typer.echo("Use --force to overwrite")
        raise typer.Exit(1)
    
    typer.echo(f"Downloading dataset: {name}")
    typer.echo(f"  Description: {dataset_info['description']}")
    typer.echo(f"  URL: {url}")
    typer.echo(f"  Expected size: {dataset_info['size_mb']} MB")
    typer.echo(f"  Output: {output_path}")
    
    try:
        _download_with_progress(url, output_path)
        typer.echo(f"✅ Download completed: {output_path}")
        
        # Verify file size
        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        typer.echo(f"Downloaded file size: {file_size_mb:.1f} MB")
        
    except Exception as e:
        typer.echo(f"❌ Download failed: {e}", err=True)
        if output_path.exists():
            output_path.unlink()  # Clean up partial download
        raise typer.Exit(1)


@app.command()
def info(
    dataset_path: Path = typer.Argument(..., help="Path to dataset file"),
):
    """Display information about a dataset."""
    if not dataset_path.exists():
        typer.echo(f"❌ Dataset file not found: {dataset_path}", err=True)
        raise typer.Exit(1)
    
    typer.echo(f"Dataset Information: {dataset_path}")
    typer.echo("=" * 50)
    
    # Basic file information
    file_size = dataset_path.stat().st_size
    typer.echo(f"File Size: {file_size / (1024*1024):.1f} MB")
    typer.echo(f"File Format: {dataset_path.suffix}")
    
    try:
        # Try to analyze the dataset based on file extension
        if dataset_path.suffix.lower() in ['.h5', '.hdf5']:
            _analyze_hdf5_dataset(dataset_path)
        elif dataset_path.suffix.lower() in ['.npy']:
            _analyze_numpy_dataset(dataset_path)
        else:
            typer.echo("⚠️  Unknown file format - cannot analyze structure")
            
    except Exception as e:
        typer.echo(f"❌ Error analyzing dataset: {e}", err=True)


def _analyze_hdf5_dataset(dataset_path: Path) -> None:
    """Analyze HDF5 dataset structure."""
    with h5py.File(dataset_path, 'r') as f:
        typer.echo(f"Format: HDF5")
        typer.echo(f"Groups: {list(f.keys())}")
        
        # Analyze each group/dataset
        for key in f.keys():
            item = f[key]
            if isinstance(item, h5py.Dataset):
                typer.echo(f"  Dataset '{key}':")
                typer.echo(f"    Shape: {item.shape}")
                typer.echo(f"    Dtype: {item.dtype}")
                typer.echo(f"    Size: {item.size:,} elements")
                
                # For vector datasets, show sample statistics
                if len(item.shape) == 2:
                    sample = item[:min(1000, item.shape[0])]
                    typer.echo(f"    Dimensions: {item.shape[1]}")
                    typer.echo(f"    Sample mean: {np.mean(sample):.4f}")
                    typer.echo(f"    Sample std: {np.std(sample):.4f}")
                    typer.echo(f"    Sample min: {np.min(sample):.4f}")
                    typer.echo(f"    Sample max: {np.max(sample):.4f}")
            
            elif isinstance(item, h5py.Group):
                typer.echo(f"  Group '{key}':")
                for subkey in item.keys():
                    subitem = item[subkey]
                    if isinstance(subitem, h5py.Dataset):
                        typer.echo(f"    Dataset '{subkey}': {subitem.shape} {subitem.dtype}")


def _analyze_numpy_dataset(dataset_path: Path) -> None:
    """Analyze numpy dataset."""
    data = np.load(dataset_path)
    
    typer.echo(f"Format: Numpy array")
    typer.echo(f"Shape: {data.shape}")
    typer.echo(f"Dtype: {data.dtype}")
    typer.echo(f"Size: {data.size:,} elements")
    
    if len(data.shape) == 2:
        typer.echo(f"Dimensions: {data.shape[1]}")
        sample = data[:min(1000, data.shape[0])]
        typer.echo(f"Sample mean: {np.mean(sample):.4f}")
        typer.echo(f"Sample std: {np.std(sample):.4f}")
        typer.echo(f"Sample min: {np.min(sample):.4f}")
        typer.echo(f"Sample max: {np.max(sample):.4f}")


@app.command()
def prepare(
    dataset_path: Path = typer.Argument(..., help="Path to raw dataset"),
    output_path: Path = typer.Argument(..., help="Output path for prepared dataset"),
    sample_size: Optional[int] = typer.Option(None, "--sample", "-s", help="Sample size"),
    format: str = typer.Option("hdf5", "--format", "-f", help="Output format (hdf5, npy)"),
):
    """Prepare a dataset for use (convert format, sample, etc.)."""
    if not dataset_path.exists():
        typer.echo(f"❌ Input dataset not found: {dataset_path}", err=True)
        raise typer.Exit(1)
    
    typer.echo(f"Preparing dataset: {dataset_path}")
    typer.echo(f"Output: {output_path}")
    
    if sample_size:
        typer.echo(f"Sample size: {sample_size:,}")
    
    try:
        # Create Dataset instance to handle the preparation
        dataset = Dataset(dataset_path=dataset_path)
        
        # For now, implement basic preparation logic
        # In a full implementation, this would handle different input formats
        typer.echo("⚠️  Dataset preparation is a placeholder")
        typer.echo("This would implement:")
        typer.echo("  - Format conversion (CSV to HDF5, etc.)")
        typer.echo("  - Data sampling and shuffling")
        typer.echo("  - Normalization and preprocessing")
        typer.echo("  - Train/test splits")
        typer.echo("  - Ground truth generation")
        
        # Create output directory
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Placeholder: copy the file for now
        import shutil
        shutil.copy2(dataset_path, output_path)
        
        typer.echo(f"✅ Dataset prepared: {output_path}")
        
    except Exception as e:
        typer.echo(f"❌ Dataset preparation failed: {e}", err=True)
        raise typer.Exit(1)


@app.command()
def list():
    """List available datasets for download."""
    typer.echo("Available Datasets:")
    typer.echo("=" * 50)
    
    for name, info in DATASETS.items():
        typer.echo(f"Name: {name}")
        typer.echo(f"  Description: {info['description']}")
        typer.echo(f"  Dimensions: {info['dimensions']}")
        typer.echo(f"  Size: {info['size_mb']} MB")
        typer.echo(f"  Format: {info['format']}")
        typer.echo()
