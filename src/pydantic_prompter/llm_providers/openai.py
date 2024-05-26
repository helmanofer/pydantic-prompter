import json
import random
from typing import List, Union

from pydantic_prompter.common import Message, logger
from pydantic_prompter.exceptions import OpenAiAuthenticationError
from pydantic_prompter.llm_providers.base import LLM


class OpenAI(LLM):
    @staticmethod
    def to_openai_format(msgs: List[Message]):
        openai_msgs = [item.model_dump() for item in msgs]
        return openai_msgs

    def debug_prompt(self, messages: List[Message], scheme: dict) -> str:
        return json.dumps(self.to_openai_format(messages), indent=4, sort_keys=True)

    @staticmethod
    def _create_schema(scheme: str) -> dict:
        if scheme == "str":
            ret = "string"
        elif scheme == "int":
            ret = "integer"
        elif scheme == "bool":
            ret = "boolean"
        elif scheme == "float":
            ret = "float"
        else:
            raise

        simple = {
            "name": "Simple",
            "parameters": {
                "properties": {"res": {"title": "Res", "type": ret}},
                "required": ["res"],
                "title": "Simple",
                "type": "object",
            },
        }
        return simple

    def call(
        self,
        messages: List[Message],
        scheme: Union[dict, None] = None,
        return_type: Union[str, None] = None,
    ) -> str:
        from openai import OpenAI, OpenAIError
        from openai import AuthenticationError, APIConnectionError

        if return_type:
            scheme = self._create_schema(return_type)

        _function_call = {
            "name": scheme["name"],
        }
        logger.debug(f"Openai Functions: \n [{scheme}]")
        logger.debug(f"Openai function_call: \n {_function_call}")
        messages_oai = self.to_openai_format(messages)
        try:
            client = OpenAI(api_key=self.settings.openai_api_key)
            chat_completion = client.chat.completions.create(
                model=self.model_name,
                messages=messages_oai,
                functions=[scheme],
                function_call=_function_call,
                temperature=random.uniform(0.3, 1.3),
            )
        except (AuthenticationError, APIConnectionError, OpenAIError) as e:
            raise OpenAiAuthenticationError(e)
        return chat_completion.choices[0].message.function_call.arguments
