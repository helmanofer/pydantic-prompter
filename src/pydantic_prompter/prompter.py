from typing import List

from jinja2 import Template
from retry import retry

from pydantic_prompter.annotation_parser import AnnotationParser
from pydantic_prompter.common import logger, Message, LLMDataAndResult
from pydantic_prompter.exceptions import (
    ArgumentError,
    BadRoleError,
    Retryable,
)
from pydantic_prompter.llm_providers import get_llm


class _Pr:
    def __init__(self, function, llm: str, model_name: str, jinja: bool):
        self.jinja = jinja
        self.function = function
        self.parser = AnnotationParser.get_parser(function)
        self.ai_provider = get_llm(provider_name=llm, model_name=model_name)
        self.llm_data = None

    @retry(tries=3, delay=1, logger=logger, exceptions=(Retryable,))
    def __call__(self, *args, **inputs):
        if args:
            raise ArgumentError("please use only kwargs")
        if not self.llm_data:
            self.llm_data = LLMDataAndResult(inputs=inputs)
            self.llm_data.messages = self._parse_docstring_to_messages(**inputs)
            logger.debug(f"Calling with prompt:\n{self.build_string(**inputs)}")
        self.llm_data.retries += 1
        self.llm_data = self.call_llm(self.llm_data)

        if self.llm_data.error:
            logger.error(f"\n\n ----> START OF ERROR % {self.llm_data.retries} <---- ")
            logger.exception(self.llm_data.error)
            logger.error(
                f"\n\nError ----> \n\n{type(self.llm_data.error)}: {self.llm_data.error}"
            )
            logger.error(f"\n\nLLM output ----> \n\n{self.llm_data.raw_result}")
            logger.error(
                f"\n\nLLM clean JSON output ----> \n\n{self.llm_data.clean_result}"
            )
            logger.error(f"\n\nPrompt ----> \n\n{self.build_string(**inputs)}")
            logger.error(f"\n\n ----> END OF ERROR <---- ")
            raise self.llm_data.error
        return self.llm_data.result

    def build_string(self, **inputs) -> str:
        msgs = self._parse_docstring_to_messages(**inputs)
        res = self.ai_provider.debug_prompt(
            msgs, self.parser.llm_schema() or self.parser.llm_return_type()
        )

        return "\n".join([f"{m.role}: {m.content}" for m in res])

    def _parse_docstring_to_messages(self, **inputs) -> List[Message]:
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

    def call_llm(self, llm_data: LLMDataAndResult) -> LLMDataAndResult:
        if self.parser.llm_schema():  # pydantic schema
            return_scheme_llm_str = self.parser.llm_schema()
            ret_str = self.ai_provider.call(
                llm_data.messages, scheme=return_scheme_llm_str
            )
        else:  # simple typings
            return_scheme_llm_str = self.parser.llm_return_type()
            ret_str = self.ai_provider.call(
                llm_data.messages, return_type=return_scheme_llm_str
            )

        llm_data.raw_result = ret_str
        # res = self.ai_provider.clean_result(ret_str)
        # llm_data.clean_result = res

        self.parser.cast_result(llm_data)
        logger.debug(f"Response from llm: \n{ret_str}")
        return llm_data


class Prompter:
    def __init__(self, ai_provider: str, model_name: str, jinja=False):
        self.model_name = model_name
        self.ai_provider = ai_provider
        self.jinja = jinja

    def __call__(self, function):
        return _Pr(
            function=function,
            jinja=self.jinja,
            llm=self.ai_provider,
            model_name=self.model_name,
        )
