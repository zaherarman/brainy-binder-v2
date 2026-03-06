from pathlib import Path
import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from .config import settings
from .ingestion.pipeline import IngestionPipeline

app = typer.Typer(name="kg-rag-pipeline", help="Privacy-first local AI knowledge assistant", add_completion=False)
console = Console()

@app.command()
def ingest(
    data_dir: Path = typer.Option(None, "--data-dir"),
    reset_index: bool = typer.Option(False, "--reset-index"),
):    
    
    """Ingest documents from a directory into the knowledge base."""

    
    console.print(Panel.fit("[bold cyan]Brainy Binder - Document Ingestion[/bold cyan]", border_style="cyan"))

    data_directory = Path(data_dir) if data_dir else settings.data_dir
    
    if not data_directory.exists():
        console.print(f"[red]Error: Data directory does not exist: {data_directory}[/red]")
        console.print("[yellow]Tip: Create the directory and add some documents first.[/yellow]")
        raise typer.Exit(code=1)

    try:
        pipeline = IngestionPipeline(data_dir=data_directory, reset_index=reset_index)
        stats = pipeline.run()

        if stats["files_processed"] > 0:
            console.print("\n[bold green]✓ Ingestion successful![/bold green]")

        else:
            console.print("\n[yellow]No new documents were ingested.[/yellow]")

    except Exception as e:
        console.print(f"\n[red]Error during ingestion: {e}[/red]")
        raise typer.Exit(code=1)

