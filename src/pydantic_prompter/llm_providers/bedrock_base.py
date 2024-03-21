import abc
import json
import random
from typing import List

from jinja2 import Template

from pydantic_prompter.common import Message, logger
from pydantic_prompter.exceptions import BedRockAuthenticationError
from pydantic_prompter.llm_providers.base import LLM


class BedRock(LLM, abc.ABC):
    @staticmethod
    def clean_result(body: str):
        body = body.split("<json_schema>")[0]
        body = body.replace("</json>", "")
        body = body.replace("</str>", "")
        body = body.replace("</int>", "")
        body = body.replace("</bool>", "")
        body = body.replace("<json>", "")
        body = body.replace("<str>", "")
        body = body.replace("<int>", "")
        body = body.replace("<bool>", "")
        body = body.replace("```", "")
        if "{" in body and "}" in body:
            left = body.find("{")
            right = body.rfind("}")
            body = body[left : right + 1]  # noqa
        return body

    @property
    @abc.abstractmethod
    def _template_path(self) -> str:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def _stop_sequence(self) -> str:
        raise NotImplementedError

    @abc.abstractmethod
    def format_messages(self, msgs: List[Message]) -> str:
        raise NotImplementedError

    def _build_prompt(self, messages: List[Message], params: dict | str):
        if "prompt_templates" not in self._template_path:
            logger.info(f"Using custom prompt from {self._template_path}")
        ant_template = open(self._template_path).read()
        if isinstance(params, dict):
            scheme_ = json.dumps(params, indent=4)
        else:
            scheme_ = params
        ant_msgs = self.format_messages(messages)
        template = Template(ant_template, keep_trailing_newline=True)
        content = template.render(schema=scheme_, question=ant_msgs).strip()
        return content

    def debug_prompt(self, messages: List[Message], scheme: dict | str) -> str:
        return self._build_prompt(messages, scheme)

    def _boto_invoke(self, body):
        try:
            logger.debug(f"Request body: \n{body}")
            import boto3

            session = boto3.Session(
                aws_access_key_id=self.settings.aws_access_key_id,
                aws_secret_access_key=self.settings.aws_secret_access_key,
                aws_session_token=self.settings.aws_session_token,
                profile_name=self.settings.aws_profile,
                region_name=self.settings.aws_default_region,
            )
            client = session.client("bedrock-runtime")
            response = client.invoke_model(
                body=body,
                modelId=self.model_name,
                accept="application/json",
                contentType="application/json",
            )
        except Exception as e:
            raise BedRockAuthenticationError(e)

        return response

    def call(
        self,
        messages: List[Message],
        scheme: dict | None = None,
        return_type: str | None = None,
    ) -> str:
        content = self._build_prompt(messages, scheme or return_type)

        body = json.dumps(
            {
                "max_tokens_to_sample": 8000,
                "prompt": content,
                "stop_sequences": [self._stop_sequence],
                "temperature": random.uniform(0, 1),
            }
        )

        response = self._boto_invoke(body)
        response_body = json.loads(response.get("body").read().decode())
        logger.info(response_body)
        return response_body.get("completion")
