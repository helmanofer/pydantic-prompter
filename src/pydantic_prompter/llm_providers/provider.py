import logging
from abc import ABC
from random import uniform
from typing import List
from pydantic_prompter.common import Message
from pydantic_prompter.llm_providers.model import Model
from pydantic_prompter.settings import Settings

logger = logging.getLogger()


class Provider(ABC):
    def __init__(self, model: Model):
        self.model = model
        self.settings = Settings()

    def debug_prompt(
        self,
        messages: List[Message],
        scheme: dict | None = None,
        return_type: str | None = None,
    ):
        content: List[Message] = self.model.build_prompt(
            messages, scheme or return_type
        )
        return content

    def call(
        self,
        messages: List[Message],
        scheme: dict | None = None,
        return_type: str | None = None,
    ) -> str:
        raise NotImplementedError

    # def clean_result(self, body: str):
    #     return self.model.clean_results(body)


class OpenAI(Provider):
    @staticmethod
    def to_openai_format(msgs: List[Message]):
        return [item.model_dump() for item in msgs]

    @staticmethod
    def _create_schema(scheme: str) -> dict:
        types = {"str": "string", "int": "integer", "bool": "boolean", "float": "float"}
        if scheme not in types:
            raise ValueError(f"Invalid scheme: {scheme}")
        return {
            "name": "Simple",
            "parameters": {
                "properties": {"res": {"title": "Res", "type": types[scheme]}},
                "required": ["res"],
                "title": "Simple",
                "type": "object",
            },
        }

    def call(
        self,
        messages: List[Message],
        scheme: dict | None = None,
        return_type: str | None = None,
    ) -> str:
        from openai import (
            OpenAI,
            OpenAIError,
            AuthenticationError,
            APIConnectionError,
        )
        from random import uniform

        if return_type:
            scheme = self._create_schema(return_type)

        _function_call = {"name": scheme["name"]}
        messages_oai = self.to_openai_format(messages)
        try:
            if self.settings.openai_api_key:
                client = OpenAI(api_key=self.settings.openai_api_key)
            elif self.settings.azure_openai_api_key:
                from openai.lib.azure import AzureOpenAI

                client = AzureOpenAI(
                    api_key=self.settings.azure_openai_api_key,
                    azure_endpoint=self.settings.azure_endpoint,
                    api_version="2024-02-01",
                )
            else:
                raise OpenAIError("No API key provided")
            chat_completion = client.chat.completions.create(
                model=self.model.model_name,
                messages=messages_oai,
                functions=[scheme],
                function_call=_function_call,
                temperature=uniform(0.3, 1.3),
            )
        except (AuthenticationError, APIConnectionError, OpenAIError) as e:
            from pydantic_prompter.exceptions import OpenAiAuthenticationError

            raise OpenAiAuthenticationError(e)
        return chat_completion.choices[0].message.function_call.arguments

    def debug_prompt(
        self,
        messages: List[Message],
        scheme: dict | None = None,
        return_type: str | None = None,
    ):
        return messages


class Bedrock(Provider):
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
                modelId=self.model.model_name,
                accept="application/json",
                contentType="application/json",
            )
        except Exception as e:
            from pydantic_prompter.exceptions import BedRockAuthenticationError

            raise e

        return response

    def debug_prompt(
        self,
        messages: List[Message],
        scheme: dict | None = None,
        return_type: str | None = None,
    ):
        content: List[Message] = self.model.build_prompt(
            messages, scheme or return_type
        )
        return content

    def call(
        self,
        messages: List[Message],
        scheme: dict | None = None,
        return_type: str | None = None,
    ) -> str:
        import json

        content: List[Message] = self.model.build_prompt(
            messages, scheme or return_type
        )

        body = self.model.bedrock_format(content)

        response = self._boto_invoke(body)
        response_body = json.loads(response.get("body").read().decode())

        logger.info(response_body)
        res = (
            response_body.get("completion")
            or response_body.get("generation")
            or response_body.get("generations", [{}])[0].get("text")
            or response_body.get("content", [{}])[0].get("text")
            or response_body.get("text")
        )
        return res


class Ollama(Provider):
    def call(
        self,
        messages: List[Message],
        scheme: dict | None = None,
        return_type: str | None = None,
    ) -> str:

        self.model.add_llama_special_tokens = False

        content: List[Message] = self.model.build_prompt(
            messages, scheme or return_type
        )
        # import ollama
        # response = ollama.chat(
        #     model=self.model.model_name,
        #     messages=[m.dict() for m in content],
        #     format="json" if scheme else "",
        #     # options={"temperature": uniform(0.3, 1.3), "top_k": 3},
        # )
        # return response["message"]["content"]

        # response = ollama.generate(
        #     model=self.model.model_name,
        #     prompt="\n".join([m.content for m in content]),
        # )
        # return response["response"]
        from openai import OpenAI

        client = OpenAI(
            base_url="http://localhost:11434/v1",
            api_key="ollama",  # required, but unused
        )

        response = client.chat.completions.create(
            model=self.model.model_name,
            messages=[m.dict() for m in content],
            temperature=uniform(0.3, 1.3),
        )
        res = response.choices[0].message.content
        return res


class Cohere(Provider):
    def call(
        self,
        messages: List[Message],
        scheme: dict | None = None,
        return_type: str | None = None,
    ) -> str:
        import random
        import cohere

        co = cohere.Client(api_key=self.settings.cohere_key)
        content = self.model.build_prompt(messages, scheme or return_type)
        content = self.model.format_messages(content)
        response = co.chat(
            message=" ",
            chat_history=content,
            temperature=random.uniform(0, 1),
        )

        answer = response.text
        return answer
