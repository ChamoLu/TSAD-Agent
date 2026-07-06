from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: Literal['user', 'assistant']
    content: str


class ChatRequest(BaseModel):
    result_id: str
    message: str
    window_index: Optional[int] = None
    model: Optional[str] = None
    history: List[ChatMessage] = Field(default_factory=list)
