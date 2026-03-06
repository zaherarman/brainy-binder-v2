from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

from src.config import settings
from src.store.neo4j import Neo4jStore
from .loaders import discover_documents, load_document
from .chunking import chunk_documents

console = Console()

class IngestionPipeline:
    def __init__(self, reset_index, data_dir=None):
        self.data_dir = data_dir or settings.data_dir
        self.neo4j_store = Neo4jStore()
        self.reset_index = reset_index # Ensures a clean ingestion state

    def run(self):
        stats = {
            "files_discovered": 0,
            "files_processed": 0,
            "files_failed": 0,
            "chunks_created": 0,
            "documents_index": 0
        }

        if self.reset_index:
            console.print("[yellow]Resetting index...[/yellow]")

            self.neo4j_store.reset()

            console.print("[green]Index reset complete![/green]")

        console.print(f"[cyan]Discovering documents in {self.data_dir}...[/cyan]")
        filepaths = discover_documents(self.data_dir)
        stats["files_discovered"] = len(filepaths)

        if not filepaths:
            console.print(f"[red]No documents found in {self.data_dir}![/red]")
            return stats
        
        console.print(f"[green]Found {len(filepaths)} files in {self.data_dir}[/green]")

        with Progress(
            SpinnerColumn(), 
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console
        ) as progress:
            task = progress.add_task("Processing documents...", total=len(filepaths))

            for filepath in filepaths:
                try:                
                    documents = load_document(filepath)

                    if not documents:
                        stats["files_failed"] += 1
                        progress.update(task, advance=1)
                        continue

                    chunks = chunk_documents(documents)
                    stats["chunks_created"] += len(chunks)

                    graph_data = self.neo4j_store.schema_inferrer(chunks)
                    self.neo4j_store.store_in_neo4j(graph_data)

                    stats["files_processed"] += 1
                    stats["documents_index"] += 1

                except Exception as e:
                    console.print(f"[red]Error processing {filepath} due to error {e}[/red]")
                    stats["files_failed"] += 1
                
                progress.update(task, advance=1)

        console.print("\n[bold green]Ingestion complete![/bold green]")
        console.print(f"   > Files discovered: {stats['files_discovered']}")
        console.print(f"   > Files processed: {stats['files_processed']}")
        console.print(f"   > Files failed: {stats['files_failed']}")
        console.print(f"   > Chunks created: {stats['chunks_created']}")
        
        return stats
    
