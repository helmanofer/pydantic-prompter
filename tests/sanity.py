def test_basic():
    from pydatic_prompter import Prompter
    from pydantic import BaseModel

    class Hi(BaseModel):
        response: str

    @Prompter(llm="openai", jinja=False, model_name="gpt-3.5-turbo")
    def funct(hello) -> Hi:
        """
        - user: say {hello}
        """

    funct.build_string(hello="hi")
