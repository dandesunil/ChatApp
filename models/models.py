from pydantic import BaseModel
from typing import List, Optional

class Chat(BaseModel):
    query: str
    answer: Optional[str] = None
    chat_history: Optional[List[str]] = []

