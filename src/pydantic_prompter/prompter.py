from typing import List, Optional, Dict

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
from pydantic_prompter.llm_providers.base import LLM


class _Pr:
    def __init__(self, function, llm: str, model_name: str, jinja: bool, model_settings: Optional[Dict] = None):
        self.jinja = jinja
        self.function = function
        self.parser = AnnotationParser.get_parser(function)
        self.llm = get_llm(llm=llm, parser=self.parser, model_name=model_name, model_settings=model_settings)

    @retry(tries=3, delay=1, logger=logger, exceptions=(Retryable,))
    def __call__(self, *args, **inputs):
        if args:
            raise ArgumentError("please use only kwargs")

        llm_data = LLMDataAndResult(inputs=inputs)
        llm_data.messages = self._parse_function_to_messages(**inputs)
        logger.debug(f"Calling with prompt:\n{self.build_string(**inputs)}")
        res: LLMDataAndResult = self.call_llm(llm_data)

        if res.error:
            logger.error(f"\n\n ----> START OF ERROR <---- ")
            logger.exception(llm_data.error)
            logger.error(f"\n\nError ----> \n\n{type(res.error)}: {res.error}")
            logger.error(f"\n\nLLM output ----> \n\n{res.raw_result}")
            logger.error(f"\n\nLLM clean output ----> \n\n{res.clean_result}")
            logger.error(f"\n\nPrompt ----> \n\n{self.build_string(**inputs)}")
            logger.error(f"\n\n ----> END OF ERROR <---- ")
            raise res.error
        return res.result

    def build_string(self, **inputs) -> str:
        msgs = self._parse_function_to_messages(**inputs)
        res = self.llm.debug_prompt(
            msgs, self.parser.llm_schema() or self.parser.llm_return_type()
        )

        return res

    def _parse_function_to_messages(self, **inputs) -> List[Message]:
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
            ret_str = self.llm.call(llm_data.messages, scheme=return_scheme_llm_str)
        else:  # simple typings
            return_scheme_llm_str = self.parser.llm_return_type()
            ret_str = self.llm.call(
                llm_data.messages, return_type=return_scheme_llm_str
            )

        llm_data.raw_result = ret_str
        res = self.llm.clean_result(ret_str)
        llm_data.clean_result = res

        self.parser.cast_result(llm_data)
        logger.debug(f"Response from llm: \n{ret_str}")
        return llm_data


class Prompter:
    def __init__(self, llm: str, model_name: str, jinja=False, model_settings: Optional[Dict] = None):
        self.model_name = model_name
        self.llm = llm
        self.jinja = jinja
        self.model_settings = model_settings

    def __call__(self, function):
        return _Pr(
            function=function,
            jinja=self.jinja,
            llm=self.llm,
            model_name=self.model_name,
            model_settings=self.model_settings,
        )