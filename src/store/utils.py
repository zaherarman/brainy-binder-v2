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

def sanitize_neo4j_properties(data: dict) -> dict:
    def clean(value):
        if value is None:
            return None

        if isinstance(value, (str, int, float, bool)):
            return value

        if isinstance(value, list):
            cleaned = [clean(v) for v in value]
            if all(isinstance(v, (str, int, float, bool)) or v is None for v in cleaned):
                return cleaned
            return [str(v) for v in cleaned]

        if isinstance(value, dict):
            return str(value)

        return str(value)

    return {k: v for k, v in ((k, clean(v)) for k, v in data.items()) if v is not None}