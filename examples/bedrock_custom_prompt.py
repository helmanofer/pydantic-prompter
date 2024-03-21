import logging
import os
from pydantic_prompter import Prompter
from pydantic import BaseModel, Field
from typing import List

logging.basicConfig(
    level=logging.INFO,
)
logging.getLogger("pydantic_prompter").setLevel(logging.DEBUG)

os.environ["TEMPLATE_PATHS__COHERE"] = "./cohere_custom.jinja"


class MyChildren(BaseModel):
    num_of_children: int
    children_names: List[str] = Field(description="The names of my children")


@Prompter(llm="bedrock", model_name="anthropic.claude-v1")
def me_and_mu_children(name) -> MyChildren:
    """
    - user: hi, my name is {name} and my children are called, aa, bb, cc
    - user: |
        how many children do I have and what's their names?
    """


print(me_and_mu_children(name="Ofer"))

# >>> DEBUG:pydantic_prompter:Using bedrock provider
#     DEBUG:pydantic_prompter:Using BedRockAnthropic provider with model anthropic.claude-v1
#     DEBUG:pydantic_prompter:Using PydanticParser
#     INFO:pydantic_prompter:Using custom prompt from ./anthropic_custom.jinja
#     DEBUG:pydantic_prompter:Calling with prompt:
#      Human: You are a REST API that answers the question contained in <qq> tags.
#     Your response should be in a JSON format which it's schema is specified in the ...
#
#     <json>
#     {
