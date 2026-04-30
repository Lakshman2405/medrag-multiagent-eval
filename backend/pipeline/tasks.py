# backend/pipeline/tasks.py

from crewai import Task


def build_tasks(patient_input, mode="medqa", options=None, iteration=1, agents=None):
    def estimate_tokens(text):
        return int(len(text.split()) * 1.3)

    # 🔹 Use passed agents (from runner)
    clarifier, rag_agent, scanner, fusion, optimizer = agents

    # 🔹 Format options nicely (IMPORTANT)
    opts_tx = ""
    if mode == "medqa" and options:
        opts_tx = "\n".join(options)

    # =========================================================
    # 🔹 TASK 1 — Clarifier
    # =========================================================
    clarifier_task = Task(
        description=f"""
            You are a clinical NLP expert.

            Summarize key clinical findings in 3-4 lines.
            Use clearly labeled structure.

            QUESTION:
            {patient_input}

            OPTIONS:
            {opts_tx}
            """,
        expected_output="Structured clinical summary",
        agent=clarifier
    )

    # =========================================================
    # 🔹 TASK 2 — RAG (ALIGNED TO TOP 3)
    # =========================================================
    rag_task = Task(
        description=f"""
            Use clinical_rag_search tool with extracted findings.

            Return:
            - Top 3 diagnoses
            - ICD-10 codes
            - Matching vs non-matching symptoms
            """,
        expected_output="Ranked diagnoses",
        agent=rag_agent
    )

    # =========================================================
    # 🔹 TASK 3 — Scanner (CRITICAL FIX: STRICT JSON)
    # =========================================================
    scanner_task = Task(
        description=f"""
                Search latest clinical evidence.

                Return STRICT JSON only:
                {{
                  "summary": "",
                  "quality": ""
                }}
                """,
        expected_output="JSON evidence summary",
        agent=scanner
    )


    # =========================================================
    # 🔥 TASK 4 — Fusion (CRITICAL FIX)
    # =========================================================
    fusion_prompt = f"""
        QUESTION:
        {patient_input}

        OPTIONS:
        {opts_tx}
        """
    print(f"\n📏 Fusion Prompt Tokens (est): {estimate_tokens(fusion_prompt)}")
    fusion_task = Task(
        description=f"""
    You are a USMLE Step 2/3 expert with strong clinical reasoning.

    QUESTION:
    {patient_input}

    OPTIONS:
    {opts_tx}

    
    You must follow the below constraint and then follow a STRICT clinical reasoning protocol. Do NOT skip steps.

    If an option includes delay for approval (court, consent, paperwork) in emergency → it is ALWAYS incorrect.

    Follow above rule as first priority, then apply clinical reasoning  protocol steps below:

    Follow this protocol:

    1. Identify question type:
    diagnosis / next step / treatment / mechanism / ethics

    2. Extract key clinical signals:
    - demographics
    - symptoms
    - critical signs (shock, bleeding, distress)
    - labs/imaging

    3. Emergency rule (highest priority):
    If life-threatening → act immediately (implied consent, no delay)

    4. Apply logic:
    - Diagnosis → match full pattern
    - Next step → unstable = treat, stable = test
    - Treatment → first-line only
    - Ethics:
        - If documented directive exists (e.g., DNR) → respect it
        - If patient has decision-making capacity → respect autonomy
        - If no known wishes AND life-threatening → emergency overrides autonomy (implied consent)

    5. Eliminate wrong options (STRICT):

    Immediately eliminate any option that:
    - delays life-saving treatment (e.g., waiting for court, consent, approval)
    - prioritizes legal process over urgent care
    - ignores hemodynamic instability or distress
    - is not first-line management

    In emergencies:
    → delay = WRONG
    → legal approval = NOT required

    6. Choose SINGLE best answer

    --------------------------------------------------

    RESPOND STRICTLY IN THIS FORMAT:

    FINAL ANSWER: <A/B/C/D>

    EXPLANATION:

    Key Clinical Clues:
    - <finding 1>
    - <finding 2>
    - <finding 3>

    Clinical Interpretation:
    <what these clues indicate medically>

    Why Correct Answer:
    <clear reasoning>

    Option Elimination:

    A:
    - <why wrong>

    B:
    - <why wrong>

    C:
    - <why wrong>

    D:
    - <why wrong>

    Core Concept:
    <one-line concept>

    Edge Case Note:
    <important distinction>

    CONFIDENCE: <0-1>
    CONFIRMED
    """,
        expected_output="Deep Structured explanation",
        agent=fusion
    )

    tasks = [clarifier_task, rag_task, scanner_task, fusion_task]

    # =========================================================
    # 🔥 TASK 5 — Optimizer (Iteration >1)
    # =========================================================
    if iteration > 1:
        optimizer_task = Task(
                description=f"""
            Re-evaluate carefully with deeper reasoning.

            QUESTION:
            {patient_input}

            OPTIONS:
            {opts_tx}

            Focus on:
            - Clinical clues
            - Mechanism
            - Eliminating wrong options

            RESPOND STRICTLY:

            FINAL ANSWER: <A/B/C/D>

            EXPLANATION:
            - Key clinical clues:
            - Why correct answer:
            - Why others are wrong:

            CONFIDENCE: <0-1>
            CONFIRMED
            """,
            expected_output="Refined reasoning",
            agent=optimizer
        )
        tasks.append(optimizer_task)

    return tasks
