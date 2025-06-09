"""
Valkey Memory Stress Testing CLI

Main entry point for the vst command-line tool.
"""

import typer
from typing import Optional
from pathlib import Path
import logging
import sys

from . import commands

app = typer.Typer(
    name="vst",
    help="Valkey Memory Stress Testing Tool",
    add_completion=False,
)

# Add command groups
app.add_typer(commands.run.app, name="run", help="Run stress test scenarios")
app.add_typer(commands.dataset.app, name="dataset", help="Dataset management")
app.add_typer(commands.validate.app, name="validate", help="Validate configurations")
app.add_typer(commands.info.app, name="info", help="Display system information")
app.add_typer(commands.visualize.app, name="visualize", help="Generate visualizations from scenario data")
app.add_typer(commands.prep.prep_app, name="prep", help="Dataset preparation and S3 management")


def _configure_logging(verbose: bool, quiet: bool) -> None:
    """Configure logging based on verbosity flags."""
    if quiet:
        level = logging.ERROR
    elif verbose:
        level = logging.DEBUG
    else:
        level = logging.INFO
    
    # Configure root logger
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        stream=sys.stderr
    )
    
    # Reduce noise from third-party libraries
    if not verbose:
        logging.getLogger('urllib3').setLevel(logging.WARNING)
        logging.getLogger('h5py').setLevel(logging.WARNING)
        logging.getLogger('redis').setLevel(logging.WARNING)


@app.callback()
def main(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Suppress non-error output"),
):
    """Valkey Memory Stress Testing Tool."""
    # Configure logging
    _configure_logging(verbose, quiet)
    
    # Validate conflicting options
    if verbose and quiet:
        typer.echo("Error: Cannot use both --verbose and --quiet", err=True)
        raise typer.Exit(1)


def _get_version() -> str:
    """Get package version."""
    try:
        import importlib.metadata
        return importlib.metadata.version("valkey-stress-test")
    except Exception:
        # Fallback version if package metadata is not available
        return "0.1.0"


@app.command()
def version():
    """Display version information."""
    version_str = _get_version()
    typer.echo(f"valkey-stress-test version {version_str}")
    
    # Additional version information in verbose mode
    try:
        import importlib.metadata
        import sys
        
        typer.echo(f"Python {sys.version}")
        
        # Show key dependency versions
        deps = ['redis', 'numpy', 'typer', 'pyyaml']
        typer.echo("\nKey dependencies:")
        for dep in deps:
            try:
                dep_version = importlib.metadata.version(dep)
                typer.echo(f"  {dep}: {dep_version}")
            except importlib.metadata.PackageNotFoundError:
                typer.echo(f"  {dep}: Not found")
                
    except ImportError:
        pass


if __name__ == "__main__":
    app()
