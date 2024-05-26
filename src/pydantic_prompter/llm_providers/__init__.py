from typing import Type, Dict, Union
from pydantic_prompter.annotation_parser import AnnotationParser
from pydantic_prompter.common import logger
from pydantic_prompter.llm_providers.bedrock_anthropic import BedRockAnthropic
from pydantic_prompter.llm_providers.bedrock_cohere import BedRockCohere
from pydantic_prompter.llm_providers.bedrock_llama2 import BedRockLlama2
from pydantic_prompter.llm_providers.cohere import Cohere
from pydantic_prompter.llm_providers.openai import OpenAI
from pydantic_prompter.llm_providers.base import LLM

# Mapping of llm type and model_name prefixes to their respective classes
LLM_MODEL_MAP: Dict[str, Dict[str, Type[LLM]]] = {
    "openai": {
        "default": OpenAI,
    },
    "bedrock": {
        "anthropic": BedRockAnthropic,
        "cohere": BedRockCohere,
        "meta": BedRockLlama2,
    },
    "cohere": {
        "command": Cohere,
    },
}


def get_llm(
    llm: str,
    model_name: str,
    parser: AnnotationParser,
    model_settings: Union[dict, None] = None,
) -> LLM:
    if llm not in LLM_MODEL_MAP:
        raise ValueError(f"LLM type '{llm}' is not implemented")

    # Extract the prefix from the model name. Adjust this logic as necessary.
    model_prefix = model_name.split(".")[
        0
    ]  # Extract 'anthropic' from 'anthropic.claude-3-sonnet-20240229-v1:0'

    model_class = LLM_MODEL_MAP.get(llm, {}).get(model_prefix, None)

    if model_class is None:
        raise ValueError(
            f"Model prefix '{model_prefix}' for LLM type '{llm}' is not implemented"
        )

    logger.debug(f"Using {model_class.__name__} provider with model {model_name}")

    return model_class(model_name, parser, model_settings)
