import typer
import pytesseract
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

def configure_tesseract() -> None:
    """
    Configure pytesseract to find the `tesseract` binary.

    - Prefer explicit env var `TESSERACT_CMD` if set.
    - On Windows, fall back to the common installation path.
    - On macOS/Linux, rely on PATH/Homebrew/etc (no hard-coded path, no import-time crash).
    """
    env_cmd = os.getenv("TESSERACT_CMD")
    if env_cmd:
        pytesseract.pytesseract.tesseract_cmd = env_cmd
        return

    if os.name == "nt":
        default = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
        if default.exists():
            pytesseract.pytesseract.tesseract_cmd = str(default)

configure_tesseract()

from rich.console import Console
from rich.panel import Panel

from .config import settings
from .ingestion.pipeline import IngestionPipeline
from .llm.answer_engine import AnswerEngine
from .store.neo4j import Neo4jStore

app = typer.Typer(name="Brainy Binder v2", help="AI knowledge assistant", add_completion=False)
console = Console()

@app.command()
def ingest(
    data_dir: Path = typer.Option(None, "--data-dir", "-d", help="Directory containing documents to ingest"),
    reset_index: bool = typer.Option(False, "--reset-index", help="Reset Neo4j index before ingesting"),
):    
    
    """Ingest documents from a directory into the knowledge base."""

    
    console.print(Panel.fit("[bold cyan]Brainy Binder v2 - Document Ingestion[/bold cyan]", border_style="cyan"))

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

@app.command()
def query(
    question: str,
    pure_rag_search: bool,
    hybrid_search: bool
):
    try:
        engine = AnswerEngine(Neo4jStore())

        with console.status("[bold cyan]Searching and generating answer...[/bold cyan]"):
            if pure_rag_search:
                answer = engine.rag_search(question)
            elif hybrid_search:
                answer = engine.hybrid_search(question)

        console.print(Panel(answer, title="[bold green]Answer[/bold green]", border_style="green"))

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()