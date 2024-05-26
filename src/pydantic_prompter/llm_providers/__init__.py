from pydantic_prompter.common import logger
from pydantic_prompter.llm_providers.model import (
    Model,
    Llama2,
    CohereCommand,
    CohereCommandR,
    Claude,
    GPT,
)
from pydantic_prompter.llm_providers.provider import (
    Bedrock,
    Ollama,
    OpenAI,
    Provider,
    Cohere,
)


def get_llm(provider_name: str, model_name: str) -> Provider:
    if "llama2" in model_name:
        model_ = Llama2(model_name)
    elif "llama3" in model_name:
        model_ = Llama2(model_name)
    elif "command-r" in model_name:
        model_ = CohereCommandR(model_name)
    elif "command" in model_name:
        model_ = CohereCommand(model_name)
    elif "claude" in model_name:
        model_ = Claude(model_name)
    elif "gpt" in model_name.lower():
        model_ = GPT(model_name)
    else:
        raise Exception(f"Model not implemented:{model_name}")

    if provider_name == "bedrock":
        provider_ = Bedrock(model=model_)
    elif provider_name == "ollama":
        provider_ = Ollama(model=model_)
    elif provider_name == "openai":
        provider_ = OpenAI(model=model_)
    elif provider_name == "cohere":
        provider_ = Cohere(model=model_)
    else:
        raise Exception(f"Provider not implemented: {provider_name}")
    logger.debug(
        f"Using {provider_.__class__.__name__} provider with model {model_name}"
    )
    return provider_
