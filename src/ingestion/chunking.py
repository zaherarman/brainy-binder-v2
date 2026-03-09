from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from src.llm.services import get_embedder
from src.config import settings

def chunk_documents(documents):
    """
    Split documents into smaller chunks to make downstream tasks more efficent.

    Args:
        documents: List of documents to chunk
        chunk_size: Size of each chunk
        chunk_overlap: Overlap between chunks

    Returns:
        List of chunked elements with metadata
    """
    chunk_size = settings.chunk_size
    chunk_overlap = settings.chunk_overlap

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap, # Good for keeping context
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""] # Priority of splitting
    )

    chunked_docs = []

    for doc in documents:
        chunks = text_splitter.split_text(doc.page_content)

        for i, chunk in enumerate(chunks):
            chunk_metadata = doc.metadata.copy()
            chunk_metadata["chunk_index"] = i
            chunk_metadata["total_chunks"] = len(chunks)

            chunked_docs.append(Document(page_content=chunk, metadata=chunk_metadata))

    return chunked_docs

def ensure_chunk_vector_index(dimensions, neo4j_driver):
    """
    Neo4j vector indexes are created over an embedding property on a node label.
    This enables db.index.vector.queryNodes(...) against chunk embeddings
    """

    query = f"""
    CREATE VECTOR INDEX chunk_embedding_index IF NOT EXISTS
    FOR (c:Chunk) ON (c.embedding)
    OPTIONS {{
      indexConfig: {{
        `vector.dimensions`: {dimensions},
        `vector.similarity_function`: 'cosine'
      }}
    }}
    """

    with neo4j_driver.session(database=settings.NEO4J_DATABASE) as session:
        session.run(query)