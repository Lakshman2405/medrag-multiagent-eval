# backend/pipeline/llm_config.py

import os
from dotenv import load_dotenv
from crewai import LLM

# =========================================================
# 🔹 LOAD ENV VARIABLES
# =========================================================
load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# =========================================================
# 🔥 CLARIFIER (Deep reasoning)
# =========================================================

llm_clarifier = LLM(
    model="openrouter/deepseek/deepseek-chat",
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
    temperature=0.1,
    max_tokens=350,
    top_p=0.9
)

# =========================================================
# 🔥 SCANNER (Structured output)
# =========================================================

llm_scanner = LLM(
    model="openrouter/qwen/qwen-2.5-72b-instruct",
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
    temperature=0.1,
    max_tokens=1000,
    top_p=0.9
)

# =========================================================
# 🔥 RAG (Semantic reasoning)
# =========================================================

llm_rag = LLM(
    model="openrouter/anthropic/claude-3-haiku",
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
    temperature=0.2,
    max_tokens=1000
)

# =========================================================
# 🔥 FUSION (Final decision)
# =========================================================

llm_fusion = LLM(
    model="openrouter/anthropic/claude-3-haiku",
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
    temperature=0.2,
    max_tokens=1000
)

# =========================================================
# 🔥 OPTIMIZER (Correction)
# =========================================================

llm_optimizer = LLM(
    model="openrouter/anthropic/claude-3-haiku",
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
    temperature=0.2,
    max_tokens=1000
    
)

# =========================================================
# 🔥 DEBUG
# =========================================================

print("🚀 LLM CONFIG LOADED:")
print("Clarifier → DeepSeek")
print("Scanner   → Qwen 72B")
print("RAG       → Claude Haiku")
print("Fusion    → Claude Haiku")
print("Optimizer → Claude Haiku")