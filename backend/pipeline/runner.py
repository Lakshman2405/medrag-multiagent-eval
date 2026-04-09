# backend/pipeline/runner.py

import re
from crewai import Crew, Process
from .agents import make_agents
from .tasks import build_tasks

crewai_logs = []


def log_step(step):
    try:
        text = str(step)

        # Detect agent name
        if "Agent:" in text:
            agent_name = text.split("Agent:")[-1].strip()
            print(f"\n🤖 Running Agent: {agent_name}")

        # Capture token usage if present
        if "tokens" in text.lower():
            print(f"📊 Token Info: {text}")

        crewai_logs.append(text)

    except Exception as e:
        crewai_logs.append(f"Log error: {str(e)}")


# =========================================================
# 🔹 SAFE PARSERS
# =========================================================
def extract_answer(text):
    match = re.search(r"(FINAL ANSWER|BEST ANSWER):\s*([A-D])", text)
    return match.group(2) if match else "N/A"


def extract_confidence(text):
    match = re.search(r"CONFIDENCE:\s*(\d*\.?\d+)", text)

    if not match:
        return 0.5

    val = float(match.group(1))

    # 🔥 Normalize if model outputs 1–10 scale
    if val > 1:
        val = val / 10

    return round(val, 2)

# =========================================================
# 🔥 MAIN LOOP (Notebook Logic)
# =========================================================
def run_adaptive_loop(inp, mode="medqa", options=None):

    MAX_ITERS = 3
    THRESHOLD = 0.65

    final_output = ""

    for iteration in range(1, MAX_ITERS + 1):

        agents = make_agents()

        tasks = build_tasks(
            patient_input=inp,
            mode=mode,
            options=options,
            iteration=iteration,
            agents=agents
        )

        crew = Crew(
            agents=agents,
            tasks=tasks,
            process=Process.sequential,
            verbose=True,
            step_callback=log_step
        )

        try:
            print(f"\n Iteration {iteration} started")

            result = crew.kickoff()

            print(f" Iteration {iteration} completed successfully")

        except Exception as e:
            print("\nLLM FAILURE DETECTED")
            print(f"Iteration: {iteration}")
            print(f"Error: {str(e)}")

            raise e  # DO NOT suppress

        final_output = str(result)

        # 🔹 Extract confidence
        conf = extract_confidence(final_output)

        # 🔹 Stop conditions
        if "CONFIRMED" in final_output:
            return final_output, iteration

        if conf >= THRESHOLD:
            return final_output, iteration

    return final_output, MAX_ITERS


# =========================================================
# 🔹 API WRAPPERS (UNCHANGED BEHAVIOR)
# =========================================================
def run_with_terminal_logs(question, options):

    import io, sys

    log_buffer = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = log_buffer

    try:
        raw_output, iterations = run_adaptive_loop(
            inp=question,
            mode="medqa",
            options=options
        )
    except Exception as e:
        raw_output = f"FINAL ANSWER: ERROR\nCONFIDENCE: 0.0\nERROR: {str(e)}"
        iterations = 0

    sys.stdout = old_stdout
    logs = log_buffer.getvalue()

    answer = extract_answer(raw_output)
    confidence = extract_confidence(raw_output)

    return {
        "result": {
            "answer": answer,
            "confidence": confidence,
            "iterations": iterations,
            "raw_output": raw_output
        },
        "logs": logs
    }


def run_with_crewai_logs(question, options):

    global crewai_logs
    crewai_logs = []

    try:
        raw_output, iterations = run_adaptive_loop(
            inp=question,
            mode="medqa",
            options=options
        )
    except Exception as e:
        raw_output = f"FINAL ANSWER: ERROR\nCONFIDENCE: 0.0\nERROR: {str(e)}"
        iterations = 0

    answer = extract_answer(raw_output)
    confidence = extract_confidence(raw_output)

    return {
        "result": {
            "answer": answer,
            "confidence": confidence,
            "iterations": iterations,
            "raw_output": raw_output
        },
        "logs": "\n".join(crewai_logs)
    }