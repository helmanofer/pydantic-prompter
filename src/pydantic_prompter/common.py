import logging

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
