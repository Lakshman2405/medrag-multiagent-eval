# backend/pipeline/evaluator.py
from .dataset_loader import load_medqa_dataset
import time
import pandas as pd
import numpy as np
from collections import defaultdict

from .runner import run_adaptive_loop


# ---------------- PARSERS ----------------

def extract_answer(text):
    import re
    match = re.search(r"FINAL ANSWER:\s*([A-E])", text)
    return match.group(1) if match else None


def extract_confidence(text):
    import re
    match = re.search(r"CONFIDENCE:\s*(\d*\.?\d+)", text)
    return float(match.group(1)) if match else 0.0


# ---------------- FAILURE TYPE ----------------
def get_failure_type(row):
    if row["correct"]:
        return "CORRECT"
    return row["q_type"]  # same as your JSON


# ---------------- MAIN EVALUATION ----------------

def evaluate_dataset(csv_path, num_cases):
    dataset = load_medqa_dataset(csv_path, num_cases)


    results = []

    for i, item in enumerate(dataset):

        question = item["question"]
        options = item["options"]
        truth = item["answer"]
        q_type = item["q_type"]

        start = time.time()

        output, iters = run_adaptive_loop(
            inp=question,
            mode="medqa",
            options=options
        )

        end = time.time()

        pred = extract_answer(output)
        conf = extract_confidence(output)

        correct = pred == truth

        results.append({
            "case": i + 1,
            "truth": truth,
            "predicted": pred,
            "score": 1.0 if correct else 0.0,
            "method": "letter_match",
            "correct": correct,
            "iters": iters,
            "time_s": round(end - start, 2),
            "dataset": "medqa",
            "q_type": q_type,
            "failure_type": get_failure_type({
                "correct": correct,
                "q_type": q_type
            }),
            "confidence_bucket": conf
        })

    return build_dashboard_json(results)


# ---------------- JSON BUILDER ----------------

def build_dashboard_json(results):

    df = pd.DataFrame(results)

    total = len(df)
    correct = df["correct"].sum()

    # -------- SUMMARY --------
    summary = {
        "accuracy": round((correct / total) * 100, 2),
        "avg_time": round(df["time_s"].mean(), 3),
        "avg_iters": round(df["iters"].mean(), 2),
        "total_cases": total
    }

    # -------- ACCURACY BY TYPE --------
    acc_type = (df.groupby("q_type")["correct"].mean() * 100).to_dict()

    # -------- FAILURE DISTRIBUTION --------
    fail_dist = df[df["failure_type"] != "CORRECT"]["failure_type"].value_counts().to_dict()

    # -------- ITERATION DISTRIBUTION --------
    iter_dist = df["iters"].value_counts().to_dict()

    # -------- CALIBRATION (MATCH YOUR FORMAT) --------
    bins = np.linspace(0, 1, 11)
    df["bin"] = pd.cut(df["confidence_bucket"], bins)

    calib = []
    grouped = df.groupby("bin")

    for b in grouped:
        group = b[1]
        if len(group) == 0:
            acc = 0.0
        else:
            acc = group["correct"].mean()

        calib.append({
            "bin": str(b[0]),
            "accuracy": round(acc, 2)
        })

    return {
        "summary": summary,
        "results": results,
        "accuracy_by_type": acc_type,
        "failure_distribution": fail_dist,
        "iteration_distribution": iter_dist,
        "calibration": calib
    }