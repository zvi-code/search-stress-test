"""Dataset management commands."""

import typer
from pathlib import Path
from typing import Optional

app = typer.Typer()


@app.command()
def download(
    name: str = typer.Argument(..., help="Dataset name (e.g., openai-5m)"),
    output_dir: Optional[Path] = typer.Option(None, "--output", "-o", help="Output directory"),
    force: bool = typer.Option(False, "--force", "-f", help="Overwrite existing files"),
):
    """Download a dataset for testing."""
    # TODO: Map name to download URL
    # TODO: Download with progress bar
    # TODO: Verify checksum
    typer.echo(f"Downloading dataset: {name}")
    raise NotImplementedError("Dataset download not implemented")


@app.command()
def info(
    dataset_path: Path = typer.Argument(..., help="Path to dataset file"),
):
    """Display information about a dataset."""
    # TODO: Load dataset metadata
    # TODO: Display dimensions, size, etc.
    typer.echo(f"Dataset info for: {dataset_path}")
    raise NotImplementedError("Dataset info not implemented")


@app.command()
def prepare(
    dataset_path: Path = typer.Argument(..., help="Path to raw dataset"),
    output_path: Path = typer.Argument(..., help="Output path for prepared dataset"),
    sample_size: Optional[int] = typer.Option(None, "--sample", "-s", help="Sample size"),
):
    """Prepare a dataset for use (convert format, sample, etc.)."""
    # TODO: Load raw dataset
    # TODO: Convert to expected format
    # TODO: Optional sampling
    # TODO: Save prepared dataset
    typer.echo(f"Preparing dataset: {dataset_path}")
    raise NotImplementedError("Dataset preparation not implemented")
