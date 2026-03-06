import re
import json

def normalize_name(name: str) -> str:
    """
    Normalize entity names for rough matching.
    Example:
    'Dr. John Smith' -> 'john smith'
    'Smith' -> 'smith'
    """
    name = name.lower().strip()
    name = re.sub(r"\b(dr|mr|mrs|ms|prof)\.?\b", "", name)
    name = re.sub(r"[^a-z0-9\s]", "", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name

def sanitize_label(label: str) -> str:
    """
    Sanitize a string so it can safely be used as a Neo4j label.

    Neo4j labels must:
    - contain only letters, numbers, and underscores
    - not start with a number
    """

    label = label.strip()

    # Replace invalid characters with underscore
    label = re.sub(r"[^A-Za-z0-9_]", "_", label)

    # Prevent empty label
    if not label:
        label = "Entity"

    # Labels cannot start with a number
    if label[0].isdigit():
        label = f"L_{label}"

    return label

def sanitize_neo4j_properties(properties: dict) -> dict:
    """
    Convert properties into Neo4j-safe values.
    Neo4j supports only primitives and lists of primitives.
    """
    safe = {}

    for key, value in properties.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            safe[key] = value
        elif isinstance(value, list):
            if all(isinstance(item, (str, int, float, bool)) or item is None for item in value):
                safe[key] = value
            else:
                safe[key] = json.dumps(value)
        elif isinstance(value, dict):
            safe[key] = json.dumps(value)
        else:
            safe[key] = str(value)

    return safe