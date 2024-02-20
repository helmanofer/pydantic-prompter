import json
from json import JSONDecodeError
from typing import Dict, Any

import yaml
from pydantic import ValidationError, ConfigDict

from pydantic_prompter.common import logger, LLMDataAndResult
from pydantic_prompter.exceptions import (
    FailedToCastLLMResult,
)


class AnnotationParser:
    @classmethod
    def get_parser(cls, function) -> "AnnotationParser":
        from pydantic._internal._model_construction import ModelMetaclass

        return_obj = function.__annotations__.get("return", None)

        if isinstance(return_obj, ModelMetaclass):
            logger.debug("Using PydanticParser")
            return PydanticParser(function)
        else:
            raise Exception("Please make sure you annotate return type using Pydantic")

    def __init__(self, function):
        self.return_cls = function.__annotations__["return"]
        self.return_cls.model_config = ConfigDict(coerce_numbers_to_str=True)

    def llm_schema(self) -> Dict:
        raise NotImplementedError

    def cast_result(self, llm_data: LLMDataAndResult):
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
        return_scheme = self.return_cls.model_json_schema()
        return self.pydantic_schema(return_scheme)

    def cast_result(self, llm_data: LLMDataAndResult) -> LLMDataAndResult:
        try:
            j = json.loads(llm_data.clean_result, strict=False)
            res = self.return_cls(**j)
            llm_data.result = res
        except (ValidationError, JSONDecodeError) as e:
            llm_data.error = FailedToCastLLMResult(e)
        return llm_data
