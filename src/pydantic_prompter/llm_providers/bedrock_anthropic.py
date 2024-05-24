import json
import random
from typing import List, Union
from typing import List, Optional, Dict

from pydantic_prompter.common import Message, logger
from pydantic_prompter.llm_providers.bedrock_base import BedRock
from pydantic_prompter.annotation_parser import AnnotationParser


class BedRockAnthropic(BedRock):
    def __init__(
        self,
        model_name: str,
        parser: AnnotationParser,
        model_settings: Optional[Dict] = None,
    ):
        super().__init__(model_name, parser)
        self.model_settings = model_settings or {
            "temperature": random.uniform(0, 1),
            "max_tokens": 8000,
            "stop_sequences": ["Human:"],
            "anthropic_version": "bedrock-2023-05-31",
        }

    def _build_prompt(self, messages: List[Message], params: dict | str):
        return "\n".join([m.content for m in messages])

    @property
    def _template_path(self):
        path = self.settings.template_paths.anthropic.replace(
            "{prompt_paths}", self.parser.prompts_path
        )
        return path

    @property
    def _stop_sequence(self):
        return "Human:"

    def format_messages(self, msgs: List[Message]) -> str:
        return "\n".join([m.content for m in msgs])

    @staticmethod
    def fix_messages(msgs: List[dict]) -> List[dict]:
        # merge messages if roles do not alternate between "user" and "assistant"
        fixed_messages = []
        for m in msgs:
            if m["role"] == "system":
                m["role"] = "user"
            if fixed_messages and fixed_messages[-1]["role"] == m["role"]:
                fixed_messages[-1]["content"] += f'\n\n{m["content"]}'
            else:
                fixed_messages.append(m)
        return fixed_messages

    def call(
        self,
        messages: List[Message],
        scheme: Union[dict, None] = None,
        return_type: Union[str, None] = None,
    ) -> str:

        if scheme:
            system_message = f"""Act like a REST API that answers the question contained in <question> tags.
                    Your response should be within <json></json> xml tags in JSON format with the schema 
                    specified in the <json_schema> tags.
                    DO NOT add any other text other than the JSON response

                    <json_schema>
                    {json.dumps(scheme, indent=4)}
                    </json_schema>
                    """
        else:  # return_type:
            system_message = f"""Act like an answer bot that answers the question contained in <question> tags.
                    Your response should be within <{return_type}></{return_type}> xml tags in {return_type} format .
                    DO NOT add any other text other than the STRING response
                    """

        final_messages = [m.model_dump() for m in messages]
        final_messages = self.fix_messages(final_messages)

        # Ensure stop_sequences and anthropic_version are always included
        body = {
            "system": system_message,
            "messages": final_messages,
            "stop_sequences": self.model_settings.get(
                "stop_sequences", [self._stop_sequence]
            ),
            "anthropic_version": self.model_settings.get(
                "anthropic_version", "bedrock-2023-05-31"
            ),
            **self.model_settings,
        }

        response = self._boto_invoke(json.dumps(body))
        res = response.get("body").read().decode()
        response_body = json.loads(res)

        logger.info(response_body)
        return response_body.get("content")[0]["text"]
