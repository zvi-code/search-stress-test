"""
Valkey Memory Stress Testing CLI

Main entry point for the vst command-line tool.
"""

import typer
from typing import Optional
from pathlib import Path
import logging

from . import commands

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

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


@app.callback()
def main(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Suppress non-error output"),
):
    """Valkey Memory Stress Testing Tool."""
    # TODO: Configure logging based on verbosity
    # TODO: Load global configuration
    pass


@app.command()
def version():
    """Display version information."""
    # TODO: Display version from package
    typer.echo("valkey-stress-test version 0.1.0")


if __name__ == "__main__":
    app()
