import json
import random
from typing import List, Optional, Dict, Union

from pydantic_prompter.common import Message, logger
from pydantic_prompter.llm_providers.bedrock_base import BedRock
from pydantic_prompter.annotation_parser import AnnotationParser


class BedRockOpenAI(BedRock):
    def __init__(
        self,
        model_name: str,
        parser: AnnotationParser,
        model_settings: Optional[Dict] = None,
    ):
        super().__init__(model_name, parser)
        self.model_settings = model_settings or {}

    def call(
        self,
        messages: List[Message],
        scheme: Union[dict, None] = None,
        return_type: Union[str, None] = None,
    ) -> str:
        if scheme:
            system_message = f"""Act like a REST API that performs the requested operation the user asked according to guidelines provided.
                    Your response should be a valid JSON format, strictly adhering to the Pydantic schema provided in the pydantic_schema section.
                    Stick to the facts and details in the provided data, and follow the guidelines closely.
                    Respond in a structured JSON format according to the provided schema.
                    DO NOT add any other text other than the requested JSON response.

                    ## pydantic_schema:

                    {json.dumps(scheme, indent=4)}

                    """
        else:  # return_type:
            system_message = f"""Act like a REST API that performs the requested operation the user asked according to guidelines provided.
                    Your response should be according to the format requested in the return_type section.
                    Stick to the facts and details in the provided data, and follow the guidelines closely.
                    Respond in a structured JSON format according to the provided schema.
                    DO NOT add any other text other than the requested return_type response.

                    ## return_type:

                    {return_type}

"""
        final_messages = [m.model_dump() for m in messages]
        final_messages.insert(0, {"role": "system", "content": system_message})

        body = {
            "messages": final_messages,
            "max_tokens": self.model_settings.get("max_tokens", 8000),
            "temperature": self.model_settings.get("temperature", random.uniform(0, 1)),
            "stop": self.model_settings.get("stop", []),
        }

        response = self._boto_invoke(json.dumps(body))
        response_body = json.loads(response.get("body").read().decode())
        logger.info(response_body)
        return response_body["choices"][0]["message"]["content"]
