import neo4j
from src.llm.services import get_llm, get_embedder
from src.schema.schema import Entity, Relationship, KnowledgeGraph
from langchain_core.prompts import ChatPromptTemplate
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline
from src.config import settings
from src.store.utils import normalize_name, sanitize_label, sanitize_neo4j_properties
from collections import defaultdict
from langchain_core.documents import Document

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

        return kg_object

    def combine_chunk_graphs(self, chunk_graphs: list[KnowledgeGraph]) -> KnowledgeGraph:
        all_entities = []
        all_relationships = []
        
        for kg in chunk_graphs:
            all_entities.extend(kg.entities)
            all_relationships.extend(kg.relationships)
        
        return KnowledgeGraph(entities=all_entities, relationships=all_relationships)

    def entity_resolution(self, kg: KnowledgeGraph) -> KnowledgeGraph:
        """
        Current strategy:
            - Only consider entities mergeable if general_type and domain_type match (E.g., Apple tech company vs apple fruit)
            - Prefer the longest name as canonical
            - Merge exact normalized matches
            - Merge short-name matches cautiously (e.g. 'Smith' to 'Dr. John Smith')
        """
         # Group entities by semantic type first
        grouped = defaultdict(list)
        for entity in kg.entities:
            key = (entity.general_type, entity.domain_type) # Resolve only if both labels are the same 
            grouped[key].append(entity)

        # Final clean list of entities
        canonical_entities = []

        # Tracks how original names map to the chosen canonical name
        alias_to_canonical = {}

        for i, entities in grouped.items():
            resolved_for_group = []

            for entity in entities:
                entity_norm = normalize_name(entity.name)
                matched = None

                for existing in resolved_for_group:
                    existing_norm = normalize_name(existing.name)

                    # Case 1: exact normalized match
                    if entity_norm == existing_norm:
                        matched = existing
                        break

                    # Case 2: one looks like a shortened version of the other
                    # "smith" vs "john smith"
                    #! Can't handle ambiguity yet, e.g., what if different people have the same first/last name?
                    entity_parts = entity_norm.split()
                    existing_parts = existing_norm.split()

                    # Current entity is less detailed than existing
                    if len(entity_parts) == 1 and entity_parts[0] in existing_parts:
                        matched = existing
                        break
                    
                    # What if existing entity name is actually less detailed then current entity?
                    if len(existing_parts) == 1 and existing_parts[0] in entity_parts:
                        matched = existing
                        break
                
                # Keep current (new) entity as canonical
                if matched is None:
                    resolved_for_group.append(entity)
                    alias_to_canonical[entity.name] = entity.name

                else:
                    # Keep the more longer name. Assumes longer = more specific.
                    if len(entity.name) > len(matched.name):
                        old_name = matched.name
                        new_name = entity.name

                        # Replace matched canonical entity with better one
                        resolved_for_group.remove(matched)

                        #! entity.properties overrides matched.properties if any match. 
                        merged_properties = {**matched.properties, **entity.properties}
                        better_entity = Entity(
                            name=new_name,
                            general_type=entity.general_type,
                            domain_type=entity.domain_type,
                            properties=merged_properties,
                        )
                        resolved_for_group.append(better_entity)

                        # Update aliases
                        alias_to_canonical[old_name] = new_name
                        alias_to_canonical[entity.name] = new_name
                    else:
                        alias_to_canonical[entity.name] = matched.name

            canonical_entities.extend(resolved_for_group)

        # Second pass: make alias mapping transitively consistent
        # Follows alias chains until it reaches the final canonical name
        def resolve_alias(name: str) -> str:
            while name in alias_to_canonical and alias_to_canonical[name] != name:
                name = alias_to_canonical[name]
            return name

        for alias in list(alias_to_canonical.keys()):
            alias_to_canonical[alias] = resolve_alias(alias)

        # Rewrite relationships to canonical names
        resolved_relationships = []
        for rel in kg.relationships:
            new_source = alias_to_canonical.get(rel.source, rel.source)
            new_target = alias_to_canonical.get(rel.target, rel.target)

            resolved_relationships.append(
                Relationship(
                    source=new_source,
                    target=new_target,
                    domain_relation=rel.domain_relation,
                    properties=rel.properties,
                )
            )

        # Deduplicate relationships after rewrite
        deduped_relationships = []

        # Stores keys we've already seen
        seen = set()

        for rel in resolved_relationships:
            key = (rel.source, rel.target, rel.domain_relation)
            if key not in seen:
                seen.add(key)
                deduped_relationships.append(rel)

        return KnowledgeGraph(
            entities=canonical_entities,
            relationships=deduped_relationships,
        )
    
    def infer_document_knowledge_graph(self, chunks: list[Document]) -> KnowledgeGraph:
        chunk_graphs = [self.schema_inferrer(chunk.page_content) for chunk in chunks]
        raw_kg = self.combine_chunk_graphs(chunk_graphs)
        resolved_kg = self.entity_resolution(raw_kg)
        return resolved_kg

    def store_in_neo4j(self, resolved_kg: KnowledgeGraph) -> None:
        # MERGE means match or create this exact pattern
        # singular letters like n or s or t is just a variable name for the node inside the query
        # Create node, attach Entity label whose name equals $source_name. Arbitrary
        # += updates and/or adds, while = replaces the entire properties dict/
        
        # (Label1:Label2 {property1: value1, property2: value2})
        # (SourceEntity)-[:domain_relation {property1: value1, property2: value2}]->(TargetEntity)
        
        with self.neo4j_driver.session(database=settings.NEO4J_DATABASE) as session:
            
            # Create or merge nodes
            for entity in resolved_kg.entities:

                # Labels cannot be passed as Cypher params, so building them into the query string
                labels = ":".join(sanitize_label(label) for label in entity.labels)

                query = f"""
                MERGE (n:{labels} {{name: $name}})
                SET n += $properties
                """

                node_properties = sanitize_neo4j_properties({
                    "general_type": entity.general_type,
                    "domain_type": entity.domain_type,
                    **entity.properties,
                })
                
                session.run(query, {"name": entity.name.strip(), "properties": node_properties})

            # Create or merge relationships
            for relationship in resolved_kg.relationships:
                rel_type = sanitize_label(relationship.domain_relation).upper()
                
                query = f"""
                MATCH (s {{name: $source_name}})
                MATCH (t {{name: $target_name}})
                MERGE (s)-[r:`{rel_type}`]->(t)
                SET r += $properties
                """
                rel_properties = sanitize_neo4j_properties(relationship.properties)
                session.run(query, {"source_name": relationship.source.strip(), "target_name": relationship.target.strip(), "properties": rel_properties})

    def reset(self):
        with self.neo4j_driver.session(database=settings.NEO4J_DATABASE) as session:
            session.run("MATCH (n) DETACH DELETE n")