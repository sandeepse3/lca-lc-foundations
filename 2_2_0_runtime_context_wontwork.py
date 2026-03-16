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


agent = create_agent(model="gpt-5-nano", context_schema=ColourContext)

# This will not work as the agent does not have access to the context when invoking without tools
response = agent.invoke(
    {"messages": [HumanMessage(content="What is my favourite colour?")]},
    context=ColourContext(),
)

pprint(response)

