import logging
from tests.data_for_tests import *
from pydantic_prompter import Prompter
from pydantic_prompter.prompter import Message


def test_basic():
    from pydantic_prompter import Prompter
    from pydantic import BaseModel

    logging.getLogger("pydantic_prompter").setLevel(logging.DEBUG)
    logging.basicConfig(
        level=logging.DEBUG,
    )

    class Hi(BaseModel):
        response: str

    @Prompter(llm="openai", jinja=False, model_name="gpt-3.5-turbo")
    def funct(hello) -> Hi:
        """
        - user: say {hello}
        """

    funct.build_string(hello="hi")


def test_basic_2():
    from pydantic_prompter import Prompter
    from pydantic import BaseModel

    logging.getLogger("pydantic_prompter").setLevel(logging.DEBUG)
    logging.basicConfig(
        level=logging.DEBUG,
    )

    class Hi(BaseModel):
        response: str

    @Prompter(llm="openai", jinja=False, model_name="gpt-3.5-turbo")
    def funct(hello) -> Hi:
        """
        - user: say {hello}
        """

    assert (
        funct.build_string(hello="hi")
        == '[\n    {\n        "content": "say hi",\n        "role": "user"\n    }\n]'
    )


def test_pydantic():
    @Prompter(jinja=True, llm="openai", model_name="gpt-3.5-turbo")
    def bbb(name) -> Hey:
        """
        - system: you are a writer
        - user: hi, my name is {{ name }} and my children are called, aa, bb, cc
        - user:
            what is my name and my children name
        """

    expected = [
        Message(role="system", content="you are a writer"),
        Message(
            role="user",
            content="hi, my name is Ofer and my children are called, aa, bb, cc",
        ),
        Message(role="user", content="what is my name and my children name"),
    ]
    res = bbb.build_prompt(name="Ofer")
    assert res == expected


def test_generic():
    @Prompter(llm="openai", model_name="dc")
    def aaa(name) -> MyChildren:
        """
        - user: hi, my name is {name} and my children are called, aa, bb, cc
        - user:
            how many children do I have and what's their names?
        """

    expected = [
        Message(
            role="user",
            content="hi, my name is Ofer and my children are called, aa, bb, cc",
        ),
        Message(
            role="user", content="how many children do I have and what's their names?"
        ),
    ]
    res = aaa.build_prompt(name="Ofer")
    assert res == expected


def test_non_yaml():
    @Prompter(llm="openai", model_name="dontcare", jinja=True)
    def search_query(history) -> QueryGPTResponse:
        """
        {{ history }}

        - user:
            Generate a Google-like search query text encompassing all previous chat questions and answers
        """

    history = [
        "- assistant: what genre do you want to watch?",
        "- user: Comedy",
        "- assistant: do you want a movie or series?",
        "- user: Movie",
    ]
    res = search_query.build_prompt(history="\n".join(history))
    expected = [
        Message(role="assistant", content="what genre do you want to watch?"),
        Message(role="user", content="Comedy"),
        Message(role="assistant", content="do you want a movie or series?"),
        Message(role="user", content="Movie"),
        Message(
            role="user",
            content="Generate a Google-like search query "
            "text encompassing all previous chat questions and answers",
        ),
    ]
    assert res == expected


def test_complex_question():
    @Prompter(llm="openai", jinja=True, model_name="dc")
    def rank_recommendation(json_entries, query) -> RecommendationResults:
        """
        - user:
            Which of the following JSON entries fit best to the query. order by best fit descending
            Base your answer ONLY on the given YML entries, if you are not sure, or there are no entries

        - user:
            The JSON entries:
            {{ json_entries }}

        - user: Query - {{ query }}

        """

    expected = [
        Message(
            role="user",
            content="Which of the following JSON entries fit best to the query. order by best fit descending\n            Base your answer ONLY on the given YML entries, if you are not sure, or there are no entries",
        ),
        Message(
            role="user",
            content='The JSON entries:\n            [{"text": "Description: 65 million years ago, the only 2 survivors of a spaceship from Somaris\\n  that crash-landed on Earth must fend off dinosaurs and reach the escape vessel in\\n  time before an imminent asteroid strike threatens to destroy the planet.\\nactors: Heath Ledger; Maggie Gyllenhaal; Aaron Eckhart; Christian Bale; Michael Caine;\\n  Gary Oldman\\ngenre: Action; drama; Thriller; Crime\\nid: 2054798\\nname: \'65\'\\nyear: 2008", "id": "2054798", "name": "65", "score": 0.4606173621133591}, {"text": "Description: \'\\"Selma,\\" as in Alabama, the place where segregation in the South was\\n  at its worst, leading to a march that ended in violence, forcing a famous statement\\n  by President Lyndon B. Johnson that ultimately led to the signing of the Voting\\n  Rights Act.\'\\nactors: Alessandro Nivola; Tom Wilkinson; Giovanni Ribisi; Carmen Ejogo; David Oyelowo;\\n  Cuba Gooding Jr.\\ngenre: A-Z; Biography;  Drama; Movies; History\\nid: 2312551\\nname: Selma\\nyear: 2014", "id": "2312551", "name": "Selma", "score": 0.4856962682297593}, {"text": "Description: 25-year-old Dwayne McLaren, a former athlete turned auto mechanic, dreams\\n  of getting out of tiny Cut Bank, Montana the coldest town in America. But his effort\\n  to do so sets in motion a deadly series of events that change his life and the life\\n  of the town forever...\\nactors: Michael Stuhlbarg; Billy Bob Thornton; John Malkovich; Teresa Palmer; Bruce\\n  Dern; Liam Hemsworth\\ngenre: SHO Movies; A-Z; Thriller\\nid: 2297150\\nname: Cut Bank\\nyear: 2015", "id": "2297150", "name": "Cut Bank", "score": 0.4997423205948504}, {"text": "Description: 2003. As a parasitic fungal outbreak begins to ravage the country and\\n  the world, Joel Miller attempts to escape the escalating chaos with his daughter\\n  and brother. Twenty years later, a now hardened Joel and his partner Tess fight\\n  to survive under a totalitarian regime, while the insurgent Fireflies harbor a teenage\\n  girl with a unique gift.\\nactors: Christopher Heyerdahl; Gabriel Luna; John Hannah; Marcia Bennett; Nico Parker;\\n  Brad Leland; Brendan Fletcher; Bella Ramsey; Pedro Pascal; Anna Torv; Merle Dandridge;\\n  Josh Brener; Jerry Wasserman\\ngenre: fantasy-sci-fi; drama; adventure; original; horror\\nid: 2044099\\nname: The Last of Us\\nyear: 2023", "id": "2044099", "name": "The Last of Us", "score": 0.5337261602153078}, {"text": "Description: \\"15 year-old Lyz, a high school student in the French Alps, has been\\\\\\n  \\\\ accepted to a highly selective ski club whose aim is to train future professional\\\\\\n  \\\\ athletes. Taking a chance on his new recruit, Fred, ex-champion turned coach,\\\\\\n  \\\\ decides to make Lyz his shining star regardless of her lack of experience. Under\\\\\\n  \\\\ his influence, Lyz will have to endure more than the physical and emotional pressure\\\\\\n  \\\\ of the training. Will Lyz\\\\u2019s determination help her escape his grip?\\"\\nactors: \\"Ma\\\\xEFra Schmitt; Marie Denarnaud; Axel Auriant; Muriel Combeau; J\\\\xE9r\\\\xE9\\\\\\n  mie Renier; No\\\\xE9e Abita\\"\\ngenre: SHO Movies; drama; A-Z\\nid: 2297161\\nname: Slalom\\nyear: 2021", "id": "2297161", "name": "Slalom", "score": 0.537035608231449}, {"text": "Description: \'\\"The Hours\\" is the story of three women searching for more potent, meaningful\\n  lives. Each is alive at a different time and place, all are linked by their yearnings\\n  and their fears. Their stories intertwine, and finally come together in a surprising,\\n  transcendent moment of shared recognition.\'\\nactors: Nicole Kidman; Julianne Moore; Meryl Streep; Ed Harris; Toni Collette; Claire\\n  Danes\\ngenre: A-Z; drama; SHO Movies\\nid: 2296757\\nname: The Hours\\nyear: 2002", "id": "2296757", "name": "The Hours", "score": 0.549504722004741}, {"text": "Description: \\"19-year-old Linn\\\\xE9a leaves her small town in Sweden and heads for\\\\\\n  \\\\ Los Angeles with the aim of becoming the world\'s next big porn star, but the road\\\\\\n  \\\\ to her goal turns out to be bumpier than she imagined.\\"\\nactors: Evelyn Claire; Alice Grey; Michael Omelko; Sofia Kappel; Jason Toler; Revika\\n  Reustle\\ngenre: drama; A-Z; SHO Movies\\nid: 2297023\\nname: Pleasure\\nyear: 2021", "id": "2297023", "name": "Pleasure", "score": 0.5547804101867869}, {"text": "Description: 70-year-old widower Ben Whittaker has discovered that retirement isn\'t\\n  all it\'s cracked up to be. Seizing an opportunity to get back in the game, he becomes\\n  a senior intern at an online fashion site, founded and run by Jules Ostin.\\nactors: JoJo Kushner; Anne Hathaway; Rene Russo; Anders Holm; Andrew Rannells; Robert\\n  De Niro\\ngenre: Movies; Comedy; A-Z\\nid: 2313618\\nname: The Intern (Theatrical)\\nyear: 2015", "id": "2313618", "name": "The Intern (Theatrical)", "score": 0.5573590045204179}, {"text": "Description: 27 years after overcoming the malevolent supernatural entity Pennywise,\\n  the former members of the Losers\' Club, who have grown up and moved away from Derry,\\n  are brought back together by a devastating phone call.\\nactors: James Ransone; Isaiah Mustafa; Jay Ryan; James McAvoy; Bill Hader; Jessica\\n  Chastain\\ngenre: drama; horror; Adaptation; A-Z; Fantasy; Movies\\nid: 2313729\\nname: \'It: Chapter Two\'\\nyear: 2019", "id": "2313729", "name": "It: Chapter Two", "score": 0.5695301178785966}, {"text": "Description: 20 years since their first adventure, Lloyd and Harry go on a road trip\\n  to find Harry\'s newly discovered daughter, who was given up for adoption.\\nactors: Steve Tom; Rachel Melvin; Laurie Holden; Jim Carrey; Jeff Daniels; Kathleen\\n  Turner\\ngenre: Movies; Comedy; A-Z\\nid: 2312803\\nname: Dumb and Dumber To\\nyear: 2014", "id": "2312803", "name": "Dumb and Dumber To", "score": 0.5716085222118849}]',
        ),
        Message(role="user", content="Query - Martial Arts, Action"),
    ]

    res = rank_recommendation.build_prompt(json_entries=entries, query=query)
    assert res == expected
