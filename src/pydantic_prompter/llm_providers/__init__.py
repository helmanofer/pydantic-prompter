from pydantic_prompter.annotation_parser import AnnotationParser
from pydantic_prompter.common import logger
from pydantic_prompter.llm_providers.bedrock_anthropic import BedRockAnthropic
from pydantic_prompter.llm_providers.bedrock_cohere import BedRockCohere
from pydantic_prompter.llm_providers.bedrock_llama2 import BedRockLlama2
from pydantic_prompter.llm_providers.cohere import Cohere
from pydantic_prompter.llm_providers.openai import OpenAI


def get_llm(llm: str, model_name: str, parser: AnnotationParser) -> "LLM":
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
