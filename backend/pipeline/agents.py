from crewai import Agent
from .llm_config import llm_clarifier, llm_rag, llm_scanner, llm_fusion, llm_optimizer
from .tools import web_search_tool
from .tools import clinical_rag_search_tool

def make_agents():

    clarifier = Agent(
        role="Clinical Text Clarifier",
        goal=(
            "Extract the core medical question and ALL key clinical findings from this USMLE Step 2/3 vignette. "
            "Identify: (1) patient demographics, (2) chief complaint, (3) key symptoms and signs, "
            "(4) relevant labs/imaging, (5) what the question is specifically asking — "
            "diagnosis / next step / mechanism / treatment / prognosis. "
            "Output a structured 4-5 line summary (clearly labeled). Do NOT guess the answer yet."
        ),
        backstory="Senior clinical NLP expert specializing in USMLE Step 2/3 vignette analysis.",
        llm=llm_clarifier,
        verbose=False,
        allow_delegation=False,
    )

    rag_agent = Agent(
        role="Symptom RAG Analyzer",
        goal=(
            "Use clinical_rag_search with the key clinical findings from the previous task. "
            "Return top 3 matching conditions with ICD-10 codes and similarity scores. "
            "For each match, note which symptoms align and which do NOT align with the vignette. "
            "Keep output concise and clearly structured."
        ),
        backstory="Diagnostic AI using BioBERT semantic similarity over clinical knowledge base.",
        tools=[clinical_rag_search_tool],
        llm=llm_rag,
        verbose=False,
        allow_delegation=False,
    )

    scanner = Agent(
        role="Evidence-Based Web Scanner",
        goal=(
            "Search for the latest clinical guidelines for the top diagnosis from RAG. "
            "Focus on Step 2/3 tested facts: first-line treatments, diagnostic criteria, "
            "classic presentations. Return a 2-line evidence summary with quality rating 1-10. "
            "Return STRICT JSON only: {\"summary\": \"\", \"quality\": \"\"}"
        ),
        backstory="Clinical research librarian specializing in USMLE Step 2/3 evidence.",
        tools=[web_search_tool],
        llm=llm_scanner,
        verbose=False,
        allow_delegation=False,
    )

    fusion = Agent(
        role="Clinical Data Fusion Agent",
        goal=(
            "You are a USMLE Step 2/3 expert. Synthesize all evidence and select the single best answer. "
            "Always identify the question TYPE first (next step / diagnosis / mechanism / treatment / ethics). "
            "Apply the correct reasoning framework for that question type. "
            "Give final answer clearly and always end your response with CONFIRMED."
        ),
        backstory="Senior clinician and USMLE examiner. Final answer authority.",
        llm=llm_fusion,
        verbose=False,
        allow_delegation=False,
    )

    optimizer = Agent(
        role="Adaptive Query Optimizer",
        goal=(
            "The previous answer was uncertain. Re-analyze the vignette from scratch. "
            "Use the clinical reasoning rules and drug/ethics cheatsheet. "
            "Produce a definitive corrected answer. Clearly state the corrected answer "
            "and always end with CONFIRMED."
        ),
        backstory="Meta-learning controller that catches reasoning errors and corrects uncertain answers.",
        llm=llm_optimizer,
        verbose=False,
        allow_delegation=False,
    )

    return clarifier, rag_agent, scanner, fusion, optimizer

print("✅ Agent factory ready.")