# backend/pipeline/orchestrator.py

from .runner import run_adaptive_loop
from .parser import parse_output


def run_case(question, options, mode="medqa"):

    output, iters = run_adaptive_loop(
        inp=question,
        mode=mode,
        options=options
    )

    parsed = parse_output(output)

    return {
        "answer": parsed["answer"],
        "confidence": parsed["confidence"],
        "iterations": iters,
        "raw_output": parsed["raw"]
    }