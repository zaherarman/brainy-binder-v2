from pathlib import Path
from langchain_core.documents import Document
from src.config import settings
from src.llm.prompts import build_rag_prompt, build_summarization_prompt

#! Rag engine from previous iteration, change
class AnswerEngine:
    def __init__(self, chroma_store=None, llm_client=None, top_k=None):
        self.chroma_store = chroma_store or ChromaStore()
        self.llm_client = llm_client or MistralClient()
        self.top_k = top_k or settings.top_k

    def answer_question(self, question, top_k=None):
        k = top_k or self.top_k

        documents = self.chroma_store.similarity_search(question, k=k)

        if not documents:
            return ("I couldn't find any relevant information in your knowledge base to answer this question.", [])

        context_chunks = []

        for doc in documents:
            source = doc.metadata.get("source_path", "Unknown")

            if source != "Unknown":
                source = Path(source).name

            context_chunks.append({"content": doc.page_content, "source": source})

        messages = build_rag_prompt(question, context_chunks)
        answer = self.llm_client.chat(messages)

        return answer, documents

    def summarize_document(self, document_path, document_id):
        if not document_path and not document_id:
            raise ValueError("Must provide either document_path or document_id")

        with get_session() as session:
            if document_id:
                db_doc = session.query(dbDocument).filter(dbDocument.id == document_id).first()
    
            else:
                db_doc = session.query(dbDocument).filter(dbDocument.path == document_path).first()

            if not db_doc:
                raise ValueError(f"Document not found: {document_path or document_id}")

            title = db_doc.title
            doc_id = db_doc.id

        chunks = self.chroma_store.get_by_metadata(filter_dict={"document_id": doc_id}, limit=1000)

        if not chunks:
            raise ValueError(f"No indexed content found for document: {title}")

        sorted_chunks = sorted(chunks, key=lambda x: x.metadata.get("chunk_index", 0)) # Vector stores do not preseve original insertion order
        full_text = "\n\n".join(chunk.page_content for chunk in sorted_chunks)

        messages = build_summarization_prompt(full_text, title)
        summary = self.llm_client.chat(messages)

        return summary

    def get_document_info(self, document_path, document_id):
        with get_session() as session:
            if document_id:
                db_doc = session.query(dbDocument).filter(dbDocument.id == document_id).first()

            elif document_path:
                db_doc = session.query(dbDocument).filter(dbDocument.path == document_path).first()
                
            else:
                raise ValueError("Must provide either document_path or document_id")

            if not db_doc:
                return None

            return {
                "id": db_doc.id,
                "path": db_doc.path,
                "document_type": db_doc.document_type,
                "title": db_doc.title,
                "tags": db_doc.tags,
                "description": db_doc.description,
                "created_at": db_doc.created_at,
                "updated_at": db_doc.updated_at,
            }

    def list_documents(self, document_type=None, limit=100):
        with get_session() as session:
            query = session.query(dbDocument)
            
            if document_type:
                query = query.filter(dbDocument.document_type == document_type)
            
            docs = query.limit(limit).all()

            return [{
                "id": doc.id, 
                "path": doc.path, 
                "document_type": doc.document_type, 
                "title": doc.title, 
                "tags": doc.tags
                } for doc in docs]

    def ask_the_graph(question: str):
        cypher_prompt = ChatPromptTemplate.from_messages([
            ("system", CYPHER_SYSTEM_TEMPLATE),
            ("human", "{question}")
        ])
        
        graph.refresh_schema()
        
        chain = GraphCypherQAChain.from_llm(
            llm=llm, 
            graph=graph, 
            verbose=True, 
            cypher_prompt=cypher_prompt,
            allow_dangerous_requests=True
        )
        response = chain.invoke({"query": question})
        return response["result"]
    
    def visualize_results():
        query = "MATCH (n)-[r]->(m) RETURN n.name as source, type(r) as rel, m.name as target LIMIT 100"
        results = graph.query(query)
        net = Network(notebook=True, cdn_resources="remote", directed=True, bgcolor="#ffffff", font_color="black", height="600px")
        
        for res in results:
            net.add_node(res['source'], label=res['source'], color="#4287f5")
            net.add_node(res['target'], label=res['target'], color="#f54242")
            net.add_edge(res['source'], res['target'], label=res['rel'])
        
        return net.show("knowledge_graph.html")
    
    def search(self):
        "fulltext, vector, range, point, text "