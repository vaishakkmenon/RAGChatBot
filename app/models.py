from typing import Annotated
from pydantic import BaseModel, StringConstraints

class QuestionRequest(BaseModel):
    question: Annotated[str, StringConstraints(min_length=1, max_length=32768)]