# Welcome to Pydantic Prompter

This library helps you build prompts easily using Pydantic

This library is using [OpenAi function calling API](https://platform.openai.com/docs/guides/gpt/function-calling)

The library's API was inspired by [DeclarAI](https://github.com/vendi-ai/declarai)


## Usage
```py
from pydantic_prompter import Prompter
from pydantic import BaseModel, Field
from typing import List
import os


os.environ["OPENAI_API_KEY"] = "sk-...."


class RecommendedEntry(BaseModel):
    id: str
    name: str
    reason: str = Field(
        description="Why this entry fits the query", default=""
    )


class RecommendationResults(BaseModel):
    title: str
    entries: List[RecommendedEntry]
    

@Prompter(llm="openai", jinja=True, model_name="gpt-3.5-turbo-16k")
def rank_recommendation_entries(
        json_entries, user_query
) -> RecommendationResults:
    """
    - system: You are a movie ranking expert
    - user: >
        Which of the following JSON entries fit best to the query. 
        order by best fit descending
        Base your answer ONLY on the given YML entries, 
        if you are not sure, or there are no entries

    - user: >
        The JSON entries:
        {{ json_entries }}

    - user: "query: {{ user_query }}"

    """

my_entries = "[{\"text\": \"Description: Four everyday suburban guys come together as a ...."
print(rank_recommendation_entries(json_entries=my_entries, user_query="Romantic comedy"))

```
```console
>>> title='Romantic Comedy' entries=[RecommendedEntry(id='2312973', name='The Ugly Truth', reason='Romantic comedy genre')]
```

```py
print(rank_recommendation_entries.build_string(json_entries=my_entries, user_query="Romantic comedy"))
```

```console
>>> system: "You are a movie ranking expert"
    user: "Which of the following JSON entries fit best to the query. order by best fit descending Base your answer ONLY on the given YML entries, if you are not sure, or there are no entries"
    user: "The JSON entries: [{\"text\": \"Description: Four everyday suburban guys come together as an excuse to escape their\\n  humdrum lives one night a week. But when they accidentally discover that their town\\n  has become overrun with aliens posing as ordinary suburbanites, they have no choice\\n  but to save their neighborhood - and the world - from total extermination.\\nactors: Rosemarie DeWitt; Richard Ayoade; Will Forte; Vince Vaughn; Ben Stiller; Jonah\\n  Hill\\ngenre: Comedy; Sci-fi; A-Z; Movies\\nid: 2313484\\nname: The Watch\\nyear: 2012\", \"id\": \"2313484\", \"name\": \"The Watch\", \"score\": 0.6327405732246574}, {\"text\": \"Description: In New York, Felix, a neurotic news writer who just broke up with his\\n  wife, is urged by his chaotic friend Oscar, a sports journalist, to move in with\\n  him, but their lifestyles are as different as night and day are, so Felix's ideas\\n  about housekeeping soon begin to irritate Oscar.\\nactors: Walter Matthau; Carole Shelley; John Fiedler; Jack Lemmon; Herb Edelman; Monica\\n  Evans\\ngenre: SHO Movies; A-Z; Comedy\\nid: 2296870\\nname: The Odd Couple\\nyear: 1968\", \"id\": \"2296870\", \"name\": \"The Odd Couple\", \"score\": 0.6544904729325227}, {\"text\": \"Description: Bill Hicks in the height of his genius. Recorded at the Dominion Theatre\\n  in London, Hicks opens our eyes and minds to the hypocrisy and ludicrousness of\\n  the world around us.\\nactors: George Carlin; William Sadler; Joss Ackland; Keanu Reeves; Alex Winter; Pam\\n  Grier\\ngenre: A-Z; SHO Movies; Comedy\\nid: 2296762\\nname: Bill & Ted Bogus\\nyear: 1991\", \"id\": \"2296762\", \"name\": \"Bill & Ted Bogus\", \"score\": 0.6558914398998}, {\"text\": \"Description: 'A married workaholic, Michael Newman doesn''t have time for his wife\\n  and children, not if he''s to impress his ungrateful boss and earn a well-deserved\\n  promotion. So when he meets Morty, a loopy sales clerk, he gets the answer to his\\n  prayers: a magical remote that allows him to bypass life''s little distractions\\n  with increasingly hysterical results.'\\nactors: Jennifer Coolidge; Kate Beckinsale; Henry Winkler; Christopher Walken; Adam\\n  Sandler; Sean Astin\\ngenre: Comedy;  Drama;  Fantasy; Movies; A-Z\\nid: 2312613\\nname: Click\\nyear: 2006\", \"id\": \"2312613\", \"name\": \"Click\", \"score\": 0.6609314592757851}, {\"text\": \"Description: A modern-day renaissance man, Schaub is a failed footballer, failed MMA\\n  fighter and currently a failing podcast host. Now, he adds another achievement to\\n  his resume. He takes to the stage to tell his tales in his first-ever stand-up comedy\\n  special.\\nactors: Brendan Schaub\\ngenre: SHO Comedy; Comedy; Stand Up\\nid: 2297073\\nname: Brendan Schaub\\nyear: 2019\", \"id\": \"2297073\", \"name\": \"Brendan Schaub\", \"score\": 0.6626776523867243}, {\"text\": \"Description: A comedy that follows a group of friends as they navigate their way through\\n  the freedoms and responsibilities of unsupervised adulthood.\\nactors: Ryan Guzman; Tyler Hoechlin; Blake Jenner; J. Johnson; Zoey Deutch; Will Brittain\\ngenre: Comedy; Movies; A-Z\\nid: 2313778\\nname: Everybody Wants Some!!\\nyear: 2016\", \"id\": \"2313778\", \"name\": \"Everybody Wants Some!!\", \"score\": 0.6651257037967419}, {\"text\": \"Description: When they can no longer stomach the ever-growing weed of suburban crime,\\n  Jay and Silent Bob take on the mantles of costumed avengers Bluntman and Chronic,\\n  smashing the super-villains they accidentally create!  Can the Doobage Duo save\\n  their beloved Jersey 'burbs from their new arch enemies, The League of Shitters?  While\\n  clearly not the comic book movie the world wants, GROOVY MOVIE is the comic book\\n  movie the world needs!\\nactors: Shannon Elizabeth; Jeff Anderson; Will Ferrell; Eliza Dushku; Diedrich Bader;\\n  Ben Affleck\\ngenre: Comedy;  Animation; A-Z; SHO Movies\\nid: 2296812\\nname: Jay & Silent Bob\\nyear: 2001\", \"id\": \"2296812\", \"name\": \"Jay & Silent Bob\", \"score\": 0.6658473159410284}, {\"text\": \"Description: Shaun Brumder is a local surfer kid from Orange County who dreams of\\n  going to Stanford to become a writer and to get away from his dysfunctional family\\n  household. Except Shaun runs into one complication after another, starting when\\n  his application is rejected after his dim-witted guidance counselor sends in the\\n  wrong form.\\nactors: Harold Ramis; Catherine O'Hara; Colin Hanks; John Lithgow; Schuyler Fisk;\\n  Jack Black\\ngenre: A-Z;  Drama; Comedy; SHO Movies\\nid: 2296739\\nname: Orange County\\nyear: 2002\", \"id\": \"2296739\", \"name\": \"Orange County\", \"score\": 0.6735588798156301}, {\"text\": \"Description: A romantically challenged morning show producer is reluctantly embroiled\\n  in a series of outrageous tests by her chauvinistic correspondent to prove his theories\\n  on relationships and help her find love. His clever ploys, however, lead to an unexpected\\n  result.\\nactors: John Higgins; Gerard Butler; Eric Winter; Nick Searcy; Bree Turner; Katherine\\n  Heigl\\ngenre: Comedy; A-Z; Movies;  Romance\\nid: 2312973\\nname: The Ugly Truth\\nyear: 2009\", \"id\": \"2312973\", \"name\": \"The Ugly Truth\", \"score\": 0.6738055654194052}, {\"text\": \"Description: Two co-dependent high school seniors are forced to deal with separation\\n  anxiety after their plan to stage a booze-soaked party goes awry.\\nactors: Bill Hader; Michael Cera; Jonah Hill; Martha MacIsaac; Seth Rogen; Christopher\\n  Mintz-Plasse\\ngenre: Movies; Comedy; A-Z\\nid: 2312774\\nname: Superbad\\nyear: 2007\", \"id\": \"2312774\", \"name\": \"Superbad\", \"score\": 0.6758505462595684}]"
    user: "query: Romantic comedy"
```

## Best practices

Explicitly state the parameter name you want to get, in this example, `title`

```py
class RecommendationTitleResponse(BaseModel):
    title: str = Field(description="4 to 6 words title")


@Prompter(llm="openai", jinja=True, model_name="gpt-3.5-turbo-16k")
def recommendation_title(json_entries) -> RecommendationTitleResponse:
    """
    - user: >
        Based on the JSON entries, suggest a minimum 4 words and maximum 6 words title

    - user: >
        The JSON entries:
        {{ json_entries }}

    """

```
Don't do this

```py hl_lines="1"
class BaseResponse(BaseModel):
    text: str = Field(description="4 to 6 words text")


@Prompter(llm="openai", jinja=True, model_name="gpt-3.5-turbo-16k")
def recommendation_title(json_entries) -> BaseResponse:
    """
    ...
    """

```
