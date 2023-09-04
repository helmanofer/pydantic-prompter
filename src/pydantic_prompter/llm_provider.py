import json
from typing import Dict, List

from jinja2 import Template

from pydantic_prompter.common import Message, logger
from pydantic_prompter.exceptions import (
    OpenAiAuthenticationError,
    BedRockAuthenticationError,
)


class LLM:
    def __init__(self, model_name):
        from pydantic_prompter.settings import Settings

        self.settings = Settings()
        self.model_name = model_name

    def debug_prompt(self, messages: List[Message], scheme: dict):
        raise NotImplementedError

    def call(self, messages: List[Message], scheme: Dict) -> str:
        raise NotImplementedError

    @classmethod
    def get_llm(cls, llm: str, model_name: str):
        if llm == "openai":
            llm_inst = OpenAI(model_name)
        elif llm == "bedrock" and model_name.startswith("anthropic"):
            logger.debug("Using bedrock provider")
            llm_inst = BedRockAnthropic(model_name)
        else:
            raise Exception(f"Model not implemented {llm}, {model_name}")
        logger.debug(
            f"Using {llm_inst.__class__.__name__} provider with model {model_name}"
        )
        return llm_inst


class OpenAI(LLM):
    @staticmethod
    def to_openai_format(msgs: List[Message]):
        openai_msgs = [item.dict() for item in msgs]
        return openai_msgs

    def debug_prompt(self, messages: List[Message], scheme: dict) -> str:
        return "\n".join(map(str, self.to_openai_format(messages)))

    def call(self, messages: List[Message], scheme: dict) -> str:
        import openai
        from openai.error import AuthenticationError

        _function_call = {
            "name": scheme["name"],
        }
        logger.debug(f"Openai Functions: \n [{scheme}]")
        logger.debug(f"Openai function_call: \n {_function_call}")
        messages = self.to_openai_format(messages)
        openai.api_key = self.settings.openai_api_key
        try:
            chat_completion = openai.ChatCompletion.create(
                model=self.model_name,
                messages=messages,
                functions=[scheme],
                function_call=_function_call,
            )
        except AuthenticationError as e:
            raise OpenAiAuthenticationError(e)
        return chat_completion.choices[0].message["function_call"]["arguments"]


class BedRockAnthropic(LLM):
    def debug_prompt(self, messages: List[Message], scheme: dict) -> str:
        return self.build_prompt(messages, scheme)

    @staticmethod
    def to_anthropic_format(msgs: List[Message]):
        role_converter = {"user": "Human", "system": "Human", "assistant": "Assistant"}
        output = []
        for msg in msgs:
            output.append(f"{role_converter[msg.role]}: {msg.content}")
        return "\n".join(output)

    def build_prompt(self, messages: List[Message], scheme: dict):
        pfp = self.settings.template_paths.anthropic
        if "prompt_templates" not in pfp:
            logger.info(f"Using custom prompt from {pfp}")
        ant_template = open(pfp).read()
        ant_scheme = json.dumps(scheme["parameters"], indent=4)
        ant_msgs = self.to_anthropic_format(messages)
        template = Template(ant_template, keep_trailing_newline=True)
        content = template.render(schema=ant_scheme, question=ant_msgs).strip()
        return content

    def call(self, messages: List[Message], scheme: dict) -> str:
        content = self.build_prompt(messages, scheme)

        body = json.dumps(
            {
                "max_tokens_to_sample": 200,
                "prompt": content,
                "stop_sequences": ["Human:"],
                "temperature": 0.7,
            }
        )
        logger.debug(f"Request body: \n{body}")
        try:
            import boto3

            session = boto3.Session(
                profile_name=self.settings.aws_profile,
                region_name=self.settings.aws_default_region,
            )
            client = session.client("bedrock")
            response = client.invoke_model(
                body=body,
                modelId=self.model_name,
                accept="application/json",
                contentType="application/json",
            )
        except Exception as e:
            raise BedRockAuthenticationError(e)

        response_body = json.loads(response.get("body").read().decode())
        logger.info(response_body)
        return response_body.get("completion")
