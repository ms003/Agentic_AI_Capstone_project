from pydantic import BaseModel
from typing import Optional

class StateSchema(BaseModel):
    user_input: str
    dept_choice: str
    route: Optional[str] = None
    message: Optional[str] = None
    answer: Optional[str] = None
    initial_response: Optional[str] = None
    improved_response: Optional[str] = None
