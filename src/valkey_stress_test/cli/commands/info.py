"""System information commands."""

import typer

app = typer.Typer()


@app.command()
def system():
    """Display system information."""
    # TODO: Show CPU info
    # TODO: Show memory info
    # TODO: Show Python version
    # TODO: Show package versions
    typer.echo("System Information:")
    raise NotImplementedError("System info not implemented")


@app.command()
def redis(
    host: str = typer.Option("localhost", "--host", "-h", help="Redis host"),
    port: int = typer.Option(6379, "--port", "-p", help="Redis port"),
):
    """Display Redis/Valkey server information."""
    # TODO: Connect to Redis
    # TODO: Get INFO output
    # TODO: Display relevant sections
    typer.echo(f"Redis info for {host}:{port}")
    raise NotImplementedError("Redis info not implemented")


@app.command()
def workloads():
    """List available workloads."""
    # TODO: Get registered workloads
    # TODO: Display with descriptions
    typer.echo("Available workloads:")
    raise NotImplementedError("Workload listing not implemented")
