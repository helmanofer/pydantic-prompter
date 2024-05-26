import random
from typing import List, Union

from pydantic_prompter.common import Message, logger
from pydantic_prompter.exceptions import CohereAuthenticationError
from pydantic_prompter.llm_providers.bedrock_cohere import BedRockCohere


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
        scheme: Union[dict, None] = None,
        return_type: Union[str, None] = None,
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
