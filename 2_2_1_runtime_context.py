# %%
from dataclasses import dataclass
from pprint import pprint

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langchain.tools import ToolRuntime, tool

load_dotenv()

# %%
@dataclass
class ColourContext:
    favourite_colour: str = "blue"
    least_favourite_colour: str = "yellow"

# ## Accessing Context
# %%
@tool
def get_favourite_colour(runtime: ToolRuntime) -> str:
    """Get the favourite colour of the user"""
    return runtime.context.favourite_colour


@tool
def get_least_favourite_colour(runtime: ToolRuntime) -> str:
    """Get the least favourite colour of the user"""
    return runtime.context.least_favourite_colour


# %%
agent = create_agent(
    model="gpt-5-nano",
    tools=[get_favourite_colour, get_least_favourite_colour],
    context_schema=ColourContext,
)

response = agent.invoke(
    {"messages": [HumanMessage(content="What is my favourite colour?")]},
    # context=ColourContext(),
    context={"favourite_colour": "red", "least_favourite_colour": "green"},
)

pprint(response)

# %%
response = agent.invoke(
    {"messages": [HumanMessage(content="What is colour that I don't like?")]},
    context=ColourContext(least_favourite_colour="green")
)

pprint(response)
# %%
