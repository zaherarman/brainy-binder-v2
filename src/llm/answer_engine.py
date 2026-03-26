from pathlib import Path
from langchain_core.documents import Document
from src.config import settings
from src.llm.services import get_llm, get_embedder
from src.llm.prompts import build_rag_prompt, build_hybrid_prompt

class AnswerEngine:
    def __init__(self, neo4jstore, top_k=None):
        self.neo4jstore = neo4jstore
        self.llm = get_llm()
        self.embedder = get_embedder() 
        self.top_k = top_k or settings.TOP_K

    def rag_search(self, query: str, top_k=None) -> str:
        query_embedding = self.embedder.embed_query(query)

        cypher = """
        CALL db.index.vector.queryNodes('chunk_embedding_index', $top_k, $embedding)
        YIELD node, score
        RETURN node.id AS chunk_id,
            node.text AS text,
            node.chunk_index AS chunk_index,
            score
        ORDER BY score DESC
        """

        with self.neo4jstore.neo4j_driver.session(database=settings.NEO4J_DATABASE) as session:
            result = session.run(cypher, {"top_k": self.top_k, "embedding": query_embedding})
            results = [dict(record) for record in result]

        context_chunks = []

        for result in results:
            context_chunks.append({
                "content": result["text"],
                "source": result["chunk_id"]
            })

        messages = build_rag_prompt(query, context_chunks)
        answer = self.llm.invoke(messages)

        return answer.content
   
    def hybrid_search(self, query: str, top_k=None) -> str:
        query_embedding = self.embedder.embed_query(query)

        # collect turns bundle into list
        cypher = """
        CALL db.index.vector.queryNodes('chunk_embedding_index', $top_k, $embedding)
        YIELD node AS chunk, score

        OPTIONAL MATCH (chunk)-[:MENTIONS]->(e:Entity)
        OPTIONAL MATCH (e)-[r]->(neighbor:Entity)
        WITH chunk, score, collect(DISTINCT e.name) AS mentioned_entities,
            collect(DISTINCT CASE
                WHEN e IS NOT NULL AND r IS NOT NULL AND neighbor IS NOT NULL
                THEN {
                    source: e.name,
                    relation: type(r),
                    target: neighbor.name
                }
            END) AS raw_graph_context
        RETURN chunk.id AS chunk_id,
            chunk.text AS text,
            score,
            mentioned_entities,
            [x IN raw_graph_context WHERE x IS NOT NULL] AS graph_context
        ORDER BY score DESC
        """

        with self.neo4jstore.neo4j_driver.session(database=settings.NEO4J_DATABASE) as session:
            result = session.run(cypher, {"top_k": self.top_k, "embedding": query_embedding})
            results =  [dict(record) for record in result]

        hybrid_results = []

        for result in results:
            hybrid_results.append({
                "content": result["text"],
                "source": result["chunk_id"],
                "graph_context": result["graph_context"]
            })

        messages = build_hybrid_prompt(query, hybrid_results)
        answer = self.llm.invoke(messages)

        return answer.content