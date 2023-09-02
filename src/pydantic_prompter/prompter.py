import logging
import os
from typing import Dict, List, Any

import yaml
from jinja2 import Template
from openai.error import AuthenticationError
from pydantic import BaseModel, ValidationError
from retry import retry

from pydantic_prompter.exceptions import (
    OpenAiAuthenticationError,
    Retryable,
    FailedToCastLLMResult, NonRetryable, )

logger = logging.getLogger()


class Message(BaseModel):
    role: str
    content: str

    def __str__(self):
        return f"{self.role}: {self.content}"


class LLM:
    def __init__(self, model_name):
        self.model_name = model_name

    def call(self, messages: List[Message], scheme: Dict) -> str:
        raise NotImplementedError


class OpenAI(LLM):
    @staticmethod
    def to_openai_format(msgs: List[Message]):
        openai_msgs = [item.dict() for item in msgs]
        return openai_msgs

    def call(self, messages: List[Message], scheme: dict) -> str:
        import openai

        _function_call = {
            "name": scheme["name"],
        }
        messages = self.to_openai_format(messages)
        openai.api_key = os.getenv("OPENAI_API_KEY")
        try:
            chat_completion = openai.ChatCompletion.create(
                model=self.model_name,
                messages=messages,
                functions=[scheme],
                function_call=_function_call,
            )
        except AuthenticationError as e:
            raise OpenAiAuthenticationError(e)
        return chat_completion.choices[0].message["function_call"]["arguments"]


class AnnotationParser:
    @classmethod
    def get_parser(cls, function) -> "AnnotationParser":
        from pydantic.main import ModelMetaclass  # noqa

        return_obj = function.__annotations__.get("return", None)

        if isinstance(return_obj, ModelMetaclass):
            return PydanticParser(function)
        else:
            raise Exception("Please make sure you annotate return type using Pydantic")

    def __init__(self, function):
        self.return_cls = function.__annotations__["return"]

    def llm_schema(self) -> Dict:
        raise NotImplementedError

    def cast_result(self, result: str):
        raise NotImplementedError


class PydanticParser(AnnotationParser):
    @staticmethod
    def pydantic_schema(schema_def: Dict[str, Any]) -> Any:
        return {
            "name": schema_def["title"],
            "description": schema_def.get("description", ""),
            "parameters": schema_def,
        }

    def llm_schema(self) -> str:
        return_scheme = self.return_cls.schema()
        return self.pydantic_schema(return_scheme)

    def cast_result(self, result: str):
        try:
            return self.return_cls.parse_raw(result)
        except ValidationError:
            raise FailedToCastLLMResult(
                f"\n\nFailed to validate JSON: \n\n{result}\n\n"
            )


class _Pr:
    def __init__(self, function, llm: LLM, jinja: bool):
        self.jinja = jinja
        self.function = function
        self.llm = llm
        self.parser = AnnotationParser.get_parser(function)

    @retry(tries=3, delay=1, logger=logger, exceptions=(Retryable,))
    def __call__(self, **inputs):
        try:
            msgs = self.build_prompt(**inputs)
            return self.call_llm(msgs)
        except (OpenAiAuthenticationError, NonRetryable):
            raise
        except Retryable:
            logger.error(f"\n\nPrompt:\n\n{self.build_string(**inputs)}")
            raise

    def build_string(self, **inputs):
        msgs = self.build_prompt(**inputs)
        return "\n".join(map(str, msgs))

    def build_prompt(self, **inputs) -> List[Message]:
        if self.jinja:
            template = Template(self.function.__doc__, keep_trailing_newline=True)
            content = template.render(**inputs)
        else:
            content = self.function.__doc__.format(**inputs)

        import re
        pattern = r'\n\s*- '
        parts = re.split(pattern, content)

        messages = []
        for content_part in parts:
            if not content_part:
                continue
            content_part = content_part.strip()
            content_part_dict = yaml.safe_load(content_part)
            role, content = list(content_part_dict.items())[0]
            messages.append(Message(role=role, content=content))
        return messages

    def call_llm(self, messages: List[Message]):
        return_scheme_llm_str = self.parser.llm_schema()

        ret_str = self.llm.call(messages, return_scheme_llm_str)
        return self.parser.cast_result(ret_str)


class Prompter:
    def __init__(self, llm: str, model_name: str, jinja=False):
        self.jinja = jinja
        if llm == "openai":
            self.llm = OpenAI(model_name)

    def __call__(self, function):
        return _Pr(function=function, jinja=self.jinja, llm=self.llm)