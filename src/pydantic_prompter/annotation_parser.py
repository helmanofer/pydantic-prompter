import abc
from contextlib import suppress
from json import JSONDecodeError
from typing import Dict, Any

from pydantic import ValidationError, ConfigDict

from pydantic_prompter.common import logger, LLMDataAndResult
from pydantic_prompter.exceptions import (
    FailedToCastLLMResult,
)


class AnnotationParser:
    @classmethod
    def get_parser(cls, function) -> "AnnotationParser":
        # noinspection PyProtectedMember
        from pydantic._internal._model_construction import ModelMetaclass

        return_obj = function.__annotations__.get("return", None)

        if isinstance(return_obj, ModelMetaclass):
            logger.debug("Using PydanticParser")
            return PydanticParser(function)

        elif return_obj in [str, int, float, bool]:
            return SimpleStringParser(function)

        else:
            raise Exception("Please make sure you annotate return type using Pydantic")

    def __init__(self, function):
        self.return_cls = function.__annotations__["return"]

    @abc.abstractmethod
    def llm_return_type(self) -> str:
        raise NotImplementedError

    @abc.abstractmethod
    def llm_schema(self) -> dict:
        raise NotImplementedError

    @abc.abstractmethod
    def cast_result(self, llm_data: LLMDataAndResult):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def prompts_path(self):
        raise NotImplementedError


class PydanticParser(AnnotationParser):
    def llm_return_type(self) -> str:
        pass

    def __init__(self, function):
        super().__init__(function)
        self.return_cls.model_config = ConfigDict(coerce_numbers_to_str=True)

    @property
    def prompts_path(self):
        return "pydantic"

    @staticmethod
    def pydantic_schema(schema_def: Dict[str, Any]) -> Any:
        return {
            "name": schema_def["title"],
            "description": schema_def.get("description", ""),
            "parameters": schema_def,
        }

    def llm_schema(self) -> dict:
        return_scheme = self.return_cls.model_json_schema(mode="serialization")
        return self.pydantic_schema(return_scheme)

    def cast_result(self, llm_data: LLMDataAndResult) -> LLMDataAndResult:
        try:
            from json_repair import json_repair

            j = json_repair.loads(llm_data.raw_result)
            llm_data.clean_json_result = j
            res = self.return_cls(**j)
            llm_data.result = res
        except (ValidationError, JSONDecodeError, RecursionError) as e:
            llm_data.error = FailedToCastLLMResult(e)
        return llm_data


class SimpleStringParser(AnnotationParser):
    def llm_schema(self) -> dict:
        pass

    @property
    def prompts_path(self):
        return "simple"

    def llm_return_type(self) -> str:
        return self.return_cls.__name__

    def clean_result(self, body: str) -> str:
        import re

        if self.return_cls == bool:
            if "true" in body.lower():
                body = "true"
            elif "false" in body.lower():
                body = "false"
        elif self.return_cls == int:
            body = re.findall(r"\d+", body)[0]
        elif self.return_cls == float:
            body = re.findall(r"\d+.\d+", body)[0]
        elif self.return_cls == str:
            body = body.strip()
        return body

    def cast_result(self, llm_data: LLMDataAndResult) -> LLMDataAndResult:
        res = None
        with suppress(Exception):
            from json_repair import json_repair

            j = json_repair.loads(llm_data.raw_result, logging=logger)
            if isinstance(j, dict):
                res = j.get("res", j)
        if not res:
            llm_data.clean_json_result = self.clean_result(llm_data.raw_result)
            res = llm_data.clean_json_result
        try:
            llm_data.result = self.return_cls(res)
        except ValueError as e:
            llm_data.error = FailedToCastLLMResult(e)
        return llm_data
