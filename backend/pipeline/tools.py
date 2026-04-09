# backend/pipeline/tools.py

from crewai.tools import tool
from .rag import embedder, qdrant, COLLECTION
import requests
import os

from pipeline.rag import clinical_rag_search as rag_search
from crewai.tools import tool   # use this if already working in your env

@tool("clinical_rag_search")
def clinical_rag_search_tool(query: str) -> str:
    """
    Semantic search over medical knowledge base using embeddings + Qdrant
    """
    try:
        return rag_search(query)   # ✅ CALL CORRECT FUNCTION

    except Exception as e:
        return f"RAG error: {str(e)}"

# =========================================================
# 🔹 2. WEB SEARCH TOOL (SERPER API)
# =========================================================
SERPER_API_KEY = os.getenv("SERPER_API_KEY")


@tool("web_search_tool")
def web_search_tool(query: str) -> str:
    """
    Performs real-time web search using Serper API
    """

    try:
        url = "https://google.serper.dev/search"

        headers = {
            "X-API-KEY": SERPER_API_KEY,
            "Content-Type": "application/json"
        }

        payload = {
            "q": query
        }

        response = requests.post(url, headers=headers, json=payload)
        data = response.json()

        # Extract top results
        results = data.get("organic", [])[:5]

        snippets = []
        for r in results:
            title = r.get("title", "")
            snippet = r.get("snippet", "")
            snippets.append(f"{title}: {snippet}")

        return "\n".join(snippets)

    except Exception as e:
        return f"Web search error: {str(e)}"