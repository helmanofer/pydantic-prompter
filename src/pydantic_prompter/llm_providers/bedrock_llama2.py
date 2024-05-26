import json
import random
from typing import List, Union

from pydantic_prompter.common import Message, logger
from pydantic_prompter.llm_providers.bedrock_base import BedRock


class BedRockLlama2(BedRock):

    @property
    def _template_path(self) -> str:
        path = self.settings.template_paths.llama2.replace(
            "{prompt_paths}", self.parser.prompts_path
        )
        return path

    @property
    def _stop_sequence(self) -> str:
        return "</s>"

    def format_messages(self, msgs: List[Message]) -> str:
        output = []
        for msg in msgs:
            if msg.role == "system":
                output.append(f"<<SYS>> {msg.content} <</SYS>>")
            if msg.role == "assistant":
                output.append(f"{msg.content}")
            if msg.role == "user":
                output.append(f"[INST] {msg.content} [/INST]")
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
                "max_gen_len": 2048,
                "prompt": content,
                "temperature": random.uniform(0, 1),
            }
        )
        response = self._boto_invoke(body)
        response_body = json.loads(response.get("body").read().decode())
        logger.debug(response_body)
        return response_body.get("generation")
