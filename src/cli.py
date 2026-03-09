import typer
import pytesseract
import os
from pathlib import Path

TESSERACT_PATH = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
TESSERACT_DIR = str(TESSERACT_PATH.parent)
TESSDATA_DIR = str(TESSERACT_PATH.parent / "tessdata")

if not TESSERACT_PATH.exists():
    raise FileNotFoundError(f"Tesseract not found at {TESSERACT_PATH}")

# Make Tesseract visible to subprocesses and OCR libraries
os.environ["PATH"] = TESSERACT_DIR + os.pathsep + os.environ.get("PATH", "")
os.environ["TESSDATA_PREFIX"] = TESSDATA_DIR

# Also configure pytesseract directly
pytesseract.pytesseract.tesseract_cmd = str(TESSERACT_PATH)

from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from .config import settings
from .ingestion.pipeline import IngestionPipeline
from .llm.answer_engine import AnswerEngine

app = typer.Typer(name="Brainy Binder v2", help="AI knowledge assistant", add_completion=False)
console = Console()

@app.command()
def ingest(
    data_dir: Path,
    reset_index: bool
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
        engine = AnswerEngine()

        with console.status("[bold cyan]Searching and generating answer...[/bold cyan]"):
            if pure_rag_search:
                answer = engine.rag_search(question)
            else:
                answer = engine.hybrid_search(question)

        console.print(Panel(answer, title="[bold green]Answer[/bold green]", border_style="green"))

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()