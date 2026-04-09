import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# 🔥 Load env variables
load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# ROOT
# =========================
@app.get("/")
def root():
    return {"message": "Live Agent API running"}


# =========================
# CREWAI RUN
# =========================
@app.post("/run-case-crewai")
def run_case_crewai(data: dict):

    try:
        # 🔥 Lazy import
        from pipeline.runner import run_with_crewai_logs

        output = run_with_crewai_logs(
            data["question"],
            data["options"]
        )
        return output

    except Exception as e:
        return {
            "result": {
                "answer": "ERROR",
                "confidence": 0.0,
                "iterations": 0,
                "raw_output": str(e)
            },
            "logs": f"❌ Backend Error: {str(e)}"
        }


# =========================
# DATASET RUN
# =========================
class DatasetRequest(BaseModel):
    num_cases: int


@app.post("/run-dataset")
def run_dataset(req: DatasetRequest):

    # 🔥 Lazy import
    from pipeline.evaluator import evaluate_dataset

    num = max(1, min(req.num_cases, 100))

    data = evaluate_dataset(
        csv_path=os.path.join(os.path.dirname(__file__), "medqa_test.csv"),
        num_cases=num
    )

    return data


# =========================
# STANDARD RUN
# =========================
@app.post("/run-case")
def run_case_api(data: dict):

    # 🔥 Lazy import
    from pipeline.runner import run_with_terminal_logs

    return run_with_terminal_logs(
        data["question"],
        data["options"]
    )
