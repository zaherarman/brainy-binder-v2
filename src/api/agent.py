import json
import logging
import os

from typing import Any, AsyncGenerator
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from src.llm.services import get_llm
from .conversation import conversation_manager

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """
You are a knowledge-base assistant backed by a Neo4j graph store.

You have access to these tools:
- rag_search: Answer questions using vector (RAG) search over ingested documents. \
Use for factual questions about the content.
- hybrid_search: Answer questions using hybrid search (vector + knowledge-graph context). \
Use when relationships between entities matter.

Use the tools to provide accurate, grounded answers. \
Prefer hybrid_search when the question involves entities, relationships, or structured data.
"""


def extract_text(content: str | list) -> str:
    if isinstance(content, str):
        return content
    
    return "\n".join(c.get("text", str(c)) if isinstance(c, dict) else str(c) for c in content)

class Agent:
    """LangChain agent wired to the MCP server."""

    def __init__(self) -> None:
        self.mcp_client: MultiServerMCPClient | None = None
        self.agent: Any = None 
        
    async def startup(self) -> None:
        """Connect to MCP server, load tools, and compile the agent graph."""
        
        FORWARDED_ENV_KEYS = {
            "PATH", "PYTHONPATH", "HOME", "TMPDIR", "TEMP", "TMP",
            "OPENAI_API_KEY", "OPENAI_MODEL",
            "NEO4J_URI", "NEO4J_USERNAME", "NEO4J_PASSWORD", "NEO4J_DATABASE",
            "OLLAMA_BASE_URL", "OLLAMA_MODEL",
            "EMBEDDING_PROVIDER", "EMBEDDING_MODEL",
            "DATA_DIR",
        }
        
        subprocess_env = {k: v for k, v in os.environ.items() if k in FORWARDED_ENV_KEYS}

        self.mcp_client = MultiServerMCPClient(
            {"logistics": {
                    "command": "python",
                    "args": ["-m", "src.mcp_server"],
                    "transport": "stdio",
                    "env": subprocess_env
                    }
                }
            )
        
        all_tools = await self.mcp_client.get_tools()
        
        # ingest_documents is only accessible via POST /api/ingest, not through chat, to prevent prompt-injection attacks that could wipe the knowledge base.
        tools = [t for t in all_tools if t.name != "ingest_documents"]
        logger.info("MCP tools loaded: %s", [t.name for t in tools])

        self.agent = create_agent(
            model=get_llm(),
            tools=tools,
            system_prompt=SYSTEM_PROMPT,
        )

    async def shutdown(self) -> None:
        """Release agent and MCP client references."""
        
        self.mcp_client = None
        self.agent = None

    def build_messages(self, session_id: str, user_message: str) -> list[BaseMessage]:
        """Reconstruct message history as LangChain message objects."""
        
        history = conversation_manager.get_history(session_id)
        messages: list[BaseMessage] = []
        
        for entry in history:
            role = entry.get("role", "")
            content = entry.get("content", "")
            
            if role == "user":
                messages.append(HumanMessage(content=content))
                
            elif role == "assistant":
                messages.append(AIMessage(content=content or ""))
                
        messages.append(HumanMessage(content=user_message))
        return messages

    async def chat(self, message: str, session_id: str) -> str:
        """
        Process a chat message and return the full response (non-streaming)"""
        
        if self.agent is None:
            raise RuntimeError("Agent not started. Call startup() first.")

        conversation_manager.add_message(session_id, "user", message)
        messages = self.build_messages(session_id, message)

        result = await self.agent.ainvoke({"messages": messages})

        ai_messages = [
            m for m in result["messages"]
            if isinstance(m, AIMessage) and not m.tool_calls
        ]
        
        answer = (
            extract_text(ai_messages[-1].content)
            if ai_messages
            else "I couldn't generate a response."
        )
        
        answer = answer or "I couldn't generate a response."

        conversation_manager.add_message(session_id, "assistant", answer)
        return answer

    async def chat_stream(self, message: str, session_id: str) -> AsyncGenerator[str, None]:
        """
        Process a chat message and stream tool-status events plus the final response."""
        
        if self.agent is None:
            raise RuntimeError("Agent not started. Call startup() first.")

        conversation_manager.add_message(session_id, "user", message)
        messages = self.build_messages(session_id, message)

        full_response = ""

        async for chunk in self.agent.astream({"messages": messages}, stream_mode="updates"):
            for node_name, update in chunk.items():
                node_messages = update.get("messages", [])

                if node_name == "tools":
                    for msg in node_messages:
                        tool_name = getattr(msg, "name", "tool")
                        yield f"data: {json.dumps({'tool': tool_name, 'status': 'completed'})}\n\n"

                elif node_name == "agent":
                    for msg in node_messages:
                        if isinstance(msg, AIMessage) and not msg.tool_calls:
                            text = extract_text(msg.content)
                            if text:
                                full_response = text
                                yield f"data: {json.dumps({'content': text})}\n\n"

        if not full_response:
            full_response = "I couldn't generate a response."
            yield f"data: {json.dumps({'content': full_response})}\n\n"

        conversation_manager.add_message(session_id, "assistant", full_response)
        yield f"data: {json.dumps({'status': 'done'})}\n\n"

agent = Agent()
