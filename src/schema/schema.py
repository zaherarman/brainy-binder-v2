from pydantic import BaseModel, Field, model_validator
from typing import List, Any, Dict

# Think of a neo4j node as the following: (Label1:Label2 {property1: value1, property2: value2})
class Entity(BaseModel):
    name: str = Field(description="Entity name, e.g. 'Apple Inc.'")

    #For semantic clasification and extraction logic. Helps with validation and entity resolution

    # For generalization across different domains
    general_type: str = Field(
        description="High-level universal category, e.g. 'Person', 'Organization', 'Location', 'Date', 'Event'."
    )

    # For specialized ontology categories for specific domains
    domain_type: str = Field(
        description="Domain-specific category, e.g. 'Disease', 'Drug', 'Procedure' if the domain was medicine."
    )

    # For querying. Graph storage layer
    labels: List[str] = Field(default_factory=list) # default_factory so every object has its own list
    
    # Extra information attached to each node. Add default_factory=Dict if you want properties to be optional, will add more nodes
    properties: Dict[str, Any] = Field(
        description="Useful attributes like description, synonyms, embedding, etc."
    )

    # After entity is created, combine general_type and domain_type into labels
    @model_validator(mode="after")
    def validate_labels(self):
        self.labels = list(set(self.labels + [self.general_type, self.domain_type]))
        return self

# Think of a neo4j edge as the following: (SourceEntity)-[:domain_relation {property1: value1, property2: value2}]->(TargetEntity) 
class Relationship(BaseModel):
    source: str = Field(description="Source entity name or canonical ID")
    target: str = Field(description="Target entity name or canonical ID")

    # Actual edge
    domain_relation: str = Field(
        description="Domain-specific relation, e.g. 'TREATS', 'CAUSES'."
    )

    # Extra information attached to the edge
    properties: Dict[str, Any] = Field(
        description="Link metadata such as confidence, evidence span, date, etc."
    )

class KnowledgeGraph(BaseModel):
    entities: List[Entity]
    relationships: List[Relationship]