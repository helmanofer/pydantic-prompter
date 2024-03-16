from pydantic_prompter import Prompter
import os

os.environ["AWS_DEFAULT_REGION"] = "us-east-1"
os.environ["AWS_ACCESS_KEY_ID"] = "..."
os.environ["AWS_SECRET_ACCESS_KEY"] = "..."
os.environ["AWS_SESSION_TOKEN"] = "..."


@Prompter(llm="cohere", model_name="command")
def me_and_mu_children(name) -> int:
    """
    - user: hi, my name is {name} and my children are called, aa, bb, cc, dd, ee
    - user: |
        how many children do I have?
    """


print(me_and_mu_children(name="Zud"))
# >>> 5
