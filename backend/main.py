# backend/main.py

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from schemas import CaseInput
from pipeline.orchestrator import run_case

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "Live Agent API running"}

import io
import sys

from pipeline.runner import run_with_crewai_logs


@app.post("/run-case-crewai")
def run_case_crewai(data: dict):
    """
    New route → returns real CrewAI logs + result
    Does NOT affect existing routes
    """

    try:
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


from pydantic import BaseModel
from pipeline.evaluator import evaluate_dataset

# main.py

class DatasetRequest(BaseModel):
    num_cases: int

@app.post("/run-dataset")
def run_dataset(req: DatasetRequest):

    num = max(1, min(req.num_cases, 100))  # safety cap

    data = evaluate_dataset(
        csv_path = os.path.join(os.path.dirname(__file__), "medqa_test.csv"),
        num_cases=num
    )

    return data



from pipeline.runner import run_with_terminal_logs

@app.post("/run-case")
def run_case(data: dict):
    return run_with_terminal_logs(
        data["question"],
        data["options"]
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
    # uvicorn main:app --reload
    # uvicorn main:app --host 0.0.0.0 --port 80

