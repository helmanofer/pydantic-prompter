from pydantic_prompter import Prompter
from pydantic import BaseModel, Field
from typing import List


class MyChildren(BaseModel):
    num_of_children: int
    children_names: List[str] = Field(description="The names of my children")


@Prompter(llm="bedrock", model_name="anthropic.claude-instant-v1")
def me_and_mu_children(name) -> MyChildren:
    """
    - user: hi, my name is {name} and my children are called, aa, bb, cc
    - user: |
        how many children do I have and what's their names?
    """


print(me_and_mu_children("Ofer"))
