import neo4j
from src.llm.services import get_llm, get_embedder
from src.schema.schema import KnowledgeGraph
from langchain_core.prompts import ChatPromptTemplate
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline
from config import settings

class Neo4jStore:
    def __init__(self):
        self.llm = get_llm()
        self.embedder = get_embedder()

        self.neo4j_driver = neo4j.GraphDatabase.driver(
            settings.NEO4J_URI,
            auth=(settings.NEO4J_USERNAME, settings.NEO4J_PASSWORD)
        )

    def schema_inferrer(self, text_chunk: str) -> KnowledgeGraph:
        # Turns into an object generator. Output matches KG Pydantic model
        structured_llm = self.llm.with_structured_output(KnowledgeGraph)

        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an experienced and detail-oriented Knowledge Graph engineer. Extract ONLY the most important entities and their direct relationships from the text."),
            ("human", "Text to process: {input}")
        ])

        chain = prompt | structured_llm
        kg_object = chain.invoke({"input": text_chunk})

        print("\nLLM GENERATED JSON:")
        print(kg_object.model_dump_json(indent=2)) 

        return kg_object

    def store_in_neo4j(kg: KnowledgeGraph):
        for relationship in kg.relationships:
            rel_type = relationship.relation.replace(' ', '_').upper()

            standard_cypher = f"""
            // MERGE means match or create this exact pattern
            // s is just a variable name for the node inside this query
            // Create node, attach Entity label whose name equals $source_name. Arbitrary
            MERGE (s:Entity {{name: $source_name}}) SET s.type = $source_type
            MERGE (t:Entity {{name: $target_name}}) SET t.type = $target_type
            MERGE (s)-[r:`{rel_type}`]->(t)
            """
            graph.query(standard_cypher, params={
                "source_name": relationship.source.name.strip().title(),
                "source_type": relationship.source.type.upper(),
                "target_name": relationship.target.name.strip().title(),
                "target_type": relationship.target.type.upper()
            })

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
    
    def reset(self):
        pass

    def search(self):
        "fulltext, vector, range, point, text "