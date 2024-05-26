from typing import List, Union

from pydantic_prompter.annotation_parser import AnnotationParser
from pydantic_prompter.common import Message


class LLM:
    @staticmethod
    def clean_result(body: str):
        return body

    def __init__(self, model_name: str, parser: AnnotationParser):
        from pydantic_prompter.settings import Settings

        self.parser: AnnotationParser = parser
        self.settings = Settings()
        self.model_name = model_name

    def debug_prompt(self, messages: List[Message], scheme: Union[dict, str]):
        raise NotImplementedError

    def call(
        self,
        messages: List[Message],
        scheme: Union[dict, None] = None,
        return_type: Union[str, None] = None,
    ) -> str:
        raise NotImplementedError
