import logging
from pydantic_prompter import Prompter
from pydantic import BaseModel

logging.basicConfig(
    level=logging.INFO,
)
logging.getLogger("pydantic_prompter").setLevel(logging.DEBUG)


class Hi(BaseModel):
    response: str


@Prompter(provider="openai", jinja=False, model_name="gpt-3.5-turbo")
def funct(hello) -> Hi:
    """
    - user: say {hello}
    """


funct(hello="hi")
