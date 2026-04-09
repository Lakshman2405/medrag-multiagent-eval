# backend/pipeline/parser.py

import re


def parse_output(output: str):

    answer_match = re.search(r"FINAL ANSWER:\s*([A-E])", output)
    confidence_match = re.search(r"CONFIDENCE:\s*(\d*\.?\d+)", output)

    answer = answer_match.group(1) if answer_match else None
    confidence = float(confidence_match.group(1)) if confidence_match else None

    return {
        "answer": answer,
        "confidence": confidence,
        "raw": output
    }

