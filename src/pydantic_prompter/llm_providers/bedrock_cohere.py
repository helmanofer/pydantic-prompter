import json
import random
from typing import List, Union

from pydantic_prompter.common import Message, logger
from pydantic_prompter.llm_providers.bedrock_base import BedRock


class BedRockCohere(BedRock):
    @property
    def _template_path(self) -> str:
        path = self.settings.template_paths.cohere.replace(
            "{prompt_paths}", self.parser.prompts_path
        )
        return path

    @property
    def _stop_sequence(self) -> str:
        return "User:"

    def format_messages(self, msgs: List[Message]) -> str:
        role_converter = {"user": "User", "system": "System", "assistant": "Chatbot"}
        output = []
        for msg in msgs:
            output.append(f"{role_converter[msg.role]}: {msg.content}")
        return "\n".join(output)

    def call(
        self,
        messages: List[Message],
        scheme: Union[dict, None] = None,
        return_type: Union[str, None] = None,
    ) -> str:
        content = self._build_prompt(messages, scheme or return_type)

        body = json.dumps(
            {
                "prompt": content,
                "stop_sequences": [self._stop_sequence],
                "temperature": random.uniform(0, 1),
            }
        )
        response = self._boto_invoke(body)

        response_body = json.loads(response.get("body").read().decode())
        logger.info(response_body)

        return response_body["generations"][0]["text"]
