import inspect
import json
import logging
from abc import ABC
from random import uniform
from typing import List

from jinja2 import Template

from pydantic_prompter.common import Message

logger = logging.getLogger()


class Model(ABC):
    system_role_supported = True
    add_llama_special_tokens = True

    def __init__(self, model_name: str):
        self.model_name = model_name

    def bedrock_format(self, msgs: List[Message]):
        raise NotImplementedError

    def clean_results(self, body):
        raise NotImplementedError

    def fix_and_merge_messages(self, msgs: List[Message]) -> List[Message]:
        # merge messages if roles do not alternate between "user" and "assistant"
        fixed_messages = []
        for m in msgs:
            if not self.system_role_supported and m.role == "system":
                m.role = "user"
            if fixed_messages and fixed_messages[-1].role == m.role:
                fixed_messages[-1].content += f"\n\n{m.content}"
            else:
                fixed_messages.append(m)
        return fixed_messages

    def system_message_jinja2(self):
        raise NotImplementedError

    def assistant_hint_jinja2(self):
        raise NotImplementedError

    def build_prompt(
        self, messages: List[Message], params: dict | str
    ) -> List[Message]:
        import json

        template = Template(self.system_message_jinja2(), keep_trailing_newline=True)

        if isinstance(params, dict):
            content = template.render(
                schema=json.dumps(params, indent=4), return_type="json"
            ).strip()

            hint = Template(self.assistant_hint_jinja2()).render(return_type="json")
        else:
            content = template.render(schema=params, return_type=params).strip()
            hint = Template(self.assistant_hint_jinja2()).render(return_type=params)

        messages.insert(0, Message(role="system", content=content))
        messages.append(Message(role="assistant", content=hint))
        messages = self.fix_and_merge_messages(messages)

        return messages


class GPT(Model):
    def system_message_jinja2(self):
        pass

    def assistant_hint_jinja2(self):
        pass

    def clean_results(self, body):
        return body


class Llama2(Model):
    def __init__(self, model_name: str):
        super().__init__(model_name)

    def system_message_jinja2(self):
        pmpt = """Act like a REST API
                {% if return_type == 'json' %}
                Your response should be within JSON markdown block in JSON format
                with the schema 
                specified in the <json_schema> tags.

                <json_schema>
                {{ schema }}
                </json_schema>
                {% else %}
                Your response should be {{ return_type }} only
                {% endif %}

                DO NOT add any other text other than the {{ return_type }} response
            """
        return pmpt

    def assistant_hint_jinja2(self):
        return """
        {% if return_type == 'json' %}
        ```{{ return_type }}
        {% else %}
        {% endif %}
        """

    def clean_results(self, body):
        body = body.replace("```", "")
        body = body.replace("```json", "")
        if "{" in body and "}" in body:
            left = body.find("{")
            right = body.rfind("}")
            body = body[left : right + 1]  # noqa
        return body

    def fix_and_merge_messages(self, msgs: List[Message]) -> List[Message]:
        msgs = super().fix_and_merge_messages(msgs)
        if self.add_llama_special_tokens:
            for msg in msgs:
                if msg.role == "system":
                    msg.content = f"<<SYS>> {msg.content} <</SYS>>"
                if msg.role == "user":
                    msg.content = f"[INST] {msg.content} [/INST]"
        return msgs

    def bedrock_format(self, msgs: List[Message]):
        final_messages = "\n".join([m.content for m in msgs])
        import json
        from random import uniform

        body = json.dumps(
            {
                "max_gen_len": 2048,
                "prompt": final_messages,
                "temperature": uniform(0, 1),
            }
        )
        return body


class CohereCommand(Model):
    def clean_results(self, body):
        body = body.replace("```", "")
        if "{" in body and "}" in body:
            left = body.find("{")
            right = body.rfind("}")
            body = body[left : right + 1]  # noqa
        return body

    def system_message_jinja2(self):
        pmp = """Act like a REST API
{% if return_type == 'json' %}
Your response should be within a JSON markdown block in JSON format 
with the schema specified in the json_schema markdown block.

```json_schema
{{ schema }}
```
{% else %}
Your response should be {{ return_type }} only
{% endif %}

DO NOT add any other text other than the JSON response
"""
        return pmp

    def assistant_hint_jinja2(self):
        return """{% if return_type == 'json' %}
                ```json
                {% else %}
                {% endif %}
                """
        # return "Chatbot: ```{{ return_type }}\n"

    def bedrock_format(self, msgs: List[Message]):
        content = self.format_messages(msgs)
        body = json.dumps(
            {
                "prompt": content,
                "stop_sequences": ["User:"],
                "temperature": uniform(0, 1),
            }
        )
        return body

    @staticmethod
    def format_messages(msgs: List[Message]) -> str:
        role_converter = {"user": "User", "system": "System", "assistant": "Chatbot"}
        output = []
        for msg in msgs:
            output.append(f"{role_converter[msg.role]}: {msg.content}")
        return "\n".join(output)


class Claude(Model):
    system_role_supported = True

    def bedrock_format(self, msgs: List[Message]):
        system_message = msgs.pop(0)
        final_messages = [m.dict() for m in msgs]
        body = json.dumps(
            {
                "system": system_message.content,
                "max_tokens": 8000,
                "messages": final_messages,
                # "stop_sequences": ["Human:"],
                "temperature": uniform(0, 1),
                "anthropic_version": "bedrock-2023-05-31",
            }
        )
        return body

    def assistant_hint_jinja2(self):
        return "assistant: <{{ return_type }}>\n"

    def system_message_jinja2(self):
        pmpt = """Act like a REST API response server
            Your response should be within <{{ return_type }}></{{ return_type }}> xml tags in {{ return_type }} format
            {% if return_type == 'json' %}
            with the schema
            specified in the <json_schema> tags.

            <json_schema>
            {{ schema }}
            </json_schema>
            {% endif %}

            DO NOT add any other text other than the {{ return_type }} response
            """
        return inspect.cleandoc(pmpt)

    def clean_results(self, body: str):
        body = body.split("<json_schema>")[0]
        body = body.replace("</json>", "")
        body = body.replace("</str>", "")
        body = body.replace("</int>", "")
        body = body.replace("</bool>", "")
        body = body.replace("<json>", "")
        body = body.replace("<str>", "")
        body = body.replace("<int>", "")
        body = body.replace("<bool>", "")
        if "{" in body and "}" in body:
            left = body.find("{")
            right = body.rfind("}")
            body = body[left : right + 1]  # noqa
        return body
