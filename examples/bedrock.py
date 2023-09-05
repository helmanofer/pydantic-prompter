from pydantic_prompter import Prompter
from pydantic import BaseModel, Field
from typing import List
import os

os.environ["AWS_DEFAULT_REGION"] = "us-east-1"
os.environ["AWS_ACCESS_KEY_ID"] = "..."
os.environ["AWS_SECRET_ACCESS_KEY"] = "..."
os.environ["AWS_SESSION_TOKEN"] = "..."


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
