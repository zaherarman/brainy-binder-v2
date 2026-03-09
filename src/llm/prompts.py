def build_chat_system_prompt():

    return """You are a helpful AI assistant with access to a personal knowledge base.

    You answer questions based on retrieved context from the user's documents (notes, PDFs, bookmarks).
    When provided with context, use it to give accurate, helpful answers. Always cite sources when possible.
    If the context doesn't contain relevant information, say so politely and offer to help in other ways.
    """

def build_rag_prompt(question, context_chunks):
    system_message = """You are a helpful AI assistant that answers questions based on a personal knowledge base.

    Your task is to provide accurate, helpful answers based on the context provided below. Follow these guidelines:
    1. Answer the question using ONLY information from the provided context
    2. If the context doesn't contain enough information, say so clearly
    3. Cite sources by mentioning the document name
    4. Be concise but thorough
    5. If multiple sources provide related information, synthesize them
    
    """

    context_text = "RETRIEVED CONTEXT\n\n"

    for i, chunk in enumerate(context_chunks, 1):
        source = chunk.get("source")
        content = chunk.get("content")
        context_text += f"[{i}] Source: {source}\n{content}\n\n"

    user_message = f"{context_text}\n## Question\n\n{question}"

    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message}
    ]

def build_hybrid_prompt(question, hybrid_results):
    system_message = """You are a helpful AI assistant that answers questions using a hybrid knowledge system that combines document retrieval and a knowledge graph.

        Your task is to provide accurate, helpful answers based on the context provided below. Follow these guidelines:
        1. Use ONLY the information provided in the context.
        2. The context contains both document excerpts and structured graph facts.
        3. Prefer graph relationships for factual accuracy.
        4. Use document excerpts for explanations and details.
        5. If the context does not contain enough information, say so clearly.
        6. Cite sources using the document name or chunk index.
        7. Be concise but thorough.
    
    """

    context_text = "RETRIEVED CONTEXT\n\n"

    for i, chunk in enumerate(hybrid_results, 1):
        source = chunk.get("source")
        content = chunk.get("content")
        graph_context = chunk.get("graph_context", "")
        context_text += f"[{i}] Source: {source}\n{content}\n\n"

        if graph_context:
            context_text += "Knowledge Graph Facts:\n"
            for fact in graph_context:
                source = fact.get("source")
                relationship = fact.get("relation")
                target = fact.get("target")

                if source and relationship and target:
                    context_text += f"({source})-[{relationship}]->({target})\n"

            context_text += "\n"

    user_message = f"{context_text}\n## Question\n\n{question}"

    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message}
    ]