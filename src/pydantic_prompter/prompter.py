from typing import List

import yaml
from jinja2 import Template
from retry import retry

from pydantic_prompter.annotation_parser import AnnotationParser
from pydantic_prompter.common import logger, Message
from pydantic_prompter.exceptions import (
    BadRoleError,
    OpenAiAuthenticationError,
    Retryable,
    NonRetryable,
)
from pydantic_prompter.llm_provider import LLM


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
            logger.debug(f"Calling with prompt:\n{self.build_string(**inputs)}")
            return self.call_llm(msgs)
        except (OpenAiAuthenticationError, NonRetryable):
            raise
        except Retryable:
            logger.error(f"\n\nPrompt:\n\n{self.build_string(**inputs)}")
            raise
        except Exception as e:
            logger.error(
                f"\n\nUnknown Error\n\nPrompt:\n\n{self.build_string(**inputs)}"
            )
            raise

    def build_string(self, **inputs):
        msgs = self.build_prompt(**inputs)
        res = self.llm.debug_prompt(msgs, self.parser.llm_schema())
        return res

    def build_prompt(self, **inputs) -> List[Message]:
        if self.jinja:
            template = Template(self.function.__doc__, keep_trailing_newline=True)
            content = template.render(**inputs)
        else:
            content = self.function.__doc__.format(**inputs)

        import re

        pattern = r"-.*?(user|system|assistant):(.*?)(?=- \w+:|\Z)"
        matches = re.findall(pattern, content, re.DOTALL | re.MULTILINE)
        result = [(m[0], m[1].strip()) for m in matches]

        messages = []

        for role, content in result:
            if role not in ["user", "system", "assistant"]:
                raise BadRoleError(f"Role {role} is not valid")
            messages.append(Message(role=role, content=content.rstrip().lstrip()))

        return messages

    def call_llm(self, messages: List[Message]):
        return_scheme_llm_str = self.parser.llm_schema()

        ret_str = self.llm.call(messages, return_scheme_llm_str)
        logger.debug(f"Response from llm: \n{ret_str}")
        return self.parser.cast_result(ret_str)


class Prompter:
    def __init__(self, llm: str, model_name: str, jinja=False):
        self.jinja = jinja
        self.llm = LLM.get_llm(llm=llm, model_name=model_name)

    def __call__(self, function):
        return _Pr(function=function, jinja=self.jinja, llm=self.llm)
