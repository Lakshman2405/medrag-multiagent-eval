from pydantic import BaseModel
from typing import List

class CaseInput(BaseModel):
    question: str
    options: List[str] = []
    mode: str = "medqa"