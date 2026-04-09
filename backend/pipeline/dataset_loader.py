# backend/pipeline/dataset_loader.py

import pandas as pd
import ast


def load_medqa_dataset(csv_path, num_cases=50):
    """
    Loads MedQA-style dataset and normalizes it into:
    {
        question: str,
        options: List[str],
        answer: str (A/B/C/D),
        q_type: str
    }
    """

    df = pd.read_csv(csv_path)

    if df.empty:
        raise ValueError(" Dataset is empty")

    df = df.sample(n=min(num_cases, len(df)), random_state=None)

    dataset = []

    for i, row in df.iterrows():

        # =========================================================
        # 🔹 QUESTION
        # =========================================================
        question = str(row.get("question", "")).strip()

        if not question:
            raise ValueError(f" Missing question at row {i}")

        # =========================================================
        # 🔹 OPTIONS (STRICT + SAFE)
        # =========================================================
        raw_options = row.get("options", None)

        if raw_options is None or pd.isna(raw_options):
            raise ValueError(f" Missing options at row {i}")

        try:
            parsed = ast.literal_eval(raw_options)

            # Case 1: dict → {"A": "...", "B": "..."}
            if isinstance(parsed, dict):
                required_keys = ["A", "B", "C", "D"]

                if not all(k in parsed for k in required_keys):
                    raise ValueError(f"Incomplete options dict at row {i}")

                options = [f"{k}. {parsed[k]}" for k in required_keys]

            # Case 2: list → ["opt1", "opt2", ...]
            elif isinstance(parsed, list):
                if len(parsed) < 4:
                    raise ValueError(f"Not enough options at row {i}")

                options = [
                    f"A. {parsed[0]}",
                    f"B. {parsed[1]}",
                    f"C. {parsed[2]}",
                    f"D. {parsed[3]}"
                ]

            else:
                raise ValueError(f"Unsupported options format at row {i}")

        except Exception as e:
            raise ValueError(f" Options parsing failed at row {i}: {e}")

        # =========================================================
        # 🔹 ANSWER (STRICT NORMALIZATION)
        # =========================================================
        idx = row.get("answer_idx", None)
        answer_raw = row.get("answer", None)

        if idx is not None and not pd.isna(idx):

            # Case: already A/B/C/D
            if isinstance(idx, str):
                idx_clean = idx.strip().upper()

                if idx_clean in ["A", "B", "C", "D"]:
                    correct = idx_clean

                elif idx_clean.isdigit():
                    num = int(idx_clean)
                    if num not in [0, 1, 2, 3]:
                        raise ValueError(f"Invalid numeric answer_idx at row {i}")
                    correct = ["A", "B", "C", "D"][num]

                else:
                    raise ValueError(f"Invalid answer_idx format at row {i}: {idx}")

            # Case: numeric
            elif isinstance(idx, (int, float)):
                num = int(idx)
                if num not in [0, 1, 2, 3]:
                    raise ValueError(f"Invalid numeric answer_idx at row {i}")
                correct = ["A", "B", "C", "D"][num]

            else:
                raise ValueError(f"Unsupported answer_idx type at row {i}")

        elif answer_raw is not None:
            correct = str(answer_raw).strip().upper()

            if correct not in ["A", "B", "C", "D"]:
                raise ValueError(f"Invalid answer format at row {i}: {answer_raw}")

        else:
            raise ValueError(f"No answer provided at row {i}")

        # =========================================================
        # 🔹 QUESTION TYPE
        # =========================================================
        q_type = str(row.get("subject_name", "OTHER")).upper()

        # =========================================================
        # 🔹 FINAL STRUCTURE
        # =========================================================
        dataset.append({
            "question": question,
            "options": options,
            "answer": correct,
            "q_type": q_type
        })

    return dataset