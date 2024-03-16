import abc
import json
import random
from typing import List

from jinja2 import Template

from pydantic_prompter.annotation_parser import AnnotationParser
from pydantic_prompter.common import Message, logger
from pydantic_prompter.exceptions import (
    CohereAuthenticationError,
    OpenAiAuthenticationError,
    BedRockAuthenticationError,
)


class LLM:
    @staticmethod
    def clean_result(body: str):
        return body

    def __init__(self, model_name: str, parser: AnnotationParser):
        from pydantic_prompter.settings import Settings

        self.parser: AnnotationParser = parser
        self.settings = Settings()
        self.model_name = model_name

    def debug_prompt(self, messages: List[Message], scheme: dict | str):
        raise NotImplementedError

    def call(
        self,
        messages: List[Message],
        scheme: dict | None = None,
        return_type: str | None = None,
    ) -> str:
        raise NotImplementedError

    @classmethod
    def get_llm(cls, llm: str, model_name: str, parser: AnnotationParser) -> "LLM":
        if llm == "openai":
            llm_inst = OpenAI(model_name, parser)
        elif llm == "bedrock" and model_name.startswith("anthropic"):
            logger.debug("Using bedrock provider with Anthropic model")
            llm_inst = BedRockAnthropic(model_name, parser)
        elif llm == "bedrock" and model_name.startswith("cohere"):
            logger.debug("Using bedrock provider with Cohere model")
            llm_inst = BedRockCohere(model_name, parser)
        elif llm == "bedrock" and model_name.startswith("meta"):
            logger.debug("Using bedrock provider with Cohere model")
            llm_inst = BedRockLlama2(model_name, parser)
        elif llm == "cohere" and model_name.startswith("command"):
            logger.debug("Using Cohere model")
            llm_inst = Cohere(model_name, parser)
        else:
            raise Exception(f"Model not implemented {llm}, {model_name}")
        logger.debug(
            f"Using {llm_inst.__class__.__name__} provider with model {model_name}"
        )
        return llm_inst


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
        scheme: dict | None = None,
        return_type: str | None = None,
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


class BedRock(LLM, abc.ABC):
    @staticmethod
    def clean_result(body: str):
        body = body.split("<json_schema>")[0]
        body = body.replace("</json>", "")
        body = body.replace("</str>", "")
        body = body.replace("</int>", "")
        body = body.replace("</bool>", "")
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


class BedRockAnthropic(BedRock):
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
        role_converter = {"user": "Human", "system": "Human", "assistant": "Assistant"}
        output = []
        for msg in msgs:
            output.append(f"{role_converter[msg.role]}: {msg.content}")
        return "\n".join(output)


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
        scheme: dict | None = None,
        return_type: str | None = None,
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
        scheme: dict | None = None,
        return_type: str | None = None,
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


class Cohere(BedRockCohere):
    def clean_result(self, body: str):
        body = super().clean_result(body)
        body = body.replace("<json>", "")
        body = body.replace("<str>", "")
        body = body.replace("<int>", "")
        body = body.replace("<bool>", "")
        return body

    def call(
        self,
        messages: List[Message],
        scheme: dict | None = None,
        return_type: str | None = None,
    ) -> str:
        try:
            import cohere

            co = cohere.Client(api_key=self.settings.cohere_key)
            content = self._build_prompt(messages, scheme or return_type)

            response = co.chat(
                message=content,
                temperature=random.uniform(0, 1),
            )
            logger.debug(f"Request body: \n{content}")

        except Exception as e:
            logger.warning(e)
            raise CohereAuthenticationError(e)

        answer = response.text
        logger.debug(f"Got answer: \n{answer}")

        return answer
