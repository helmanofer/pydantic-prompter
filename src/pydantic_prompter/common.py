import logging
from typing import Optional, List, Any, Dict

from pydantic import BaseModel

from pydantic_prompter.settings import Settings

logger = logging.getLogger("pydantic_prompter")
logger.addHandler(logging.NullHandler())

settings = Settings()


class Message(BaseModel):
    role: str
    content: str

    def __str__(self):
        return f"{self.role}: {self.content}"


class LLMDataAndResult(BaseModel):
    inputs: Dict[str, Any]
    messages: Optional[List[Message]] = None
    raw_result: Optional[str] = None
    clean_result: Optional[str] = None
    result: Optional[BaseModel] = None
    error: Optional[Any] = None
