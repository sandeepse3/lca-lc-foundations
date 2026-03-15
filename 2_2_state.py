# %%
from dotenv import load_dotenv

load_dotenv()

# %%
from langchain.agents import AgentState

class CustomState(AgentState):
    favourite_colour: str

# ## Write to state

# %%
from langchain.tools import tool, ToolRuntime
from langgraph.types import Command
from langchain.messages import ToolMessage

@tool
def update_favourite_colour(favourite_colour: str, runtime: ToolRuntime) -> Command:
    """Update the favourite colour of the user in the state once they've revealed it."""
    return Command(update={
        "favourite_colour": favourite_colour, 
        "messages": [ToolMessage("Successfully updated favourite colour", tool_call_id=runtime.tool_call_id)]}
        )

# %%
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

agent = create_agent(
    "gpt-5-nano",
    tools=[update_favourite_colour],
    checkpointer=InMemorySaver(),
    state_schema=CustomState
)

# %%
from langchain.messages import HumanMessage

response = agent.invoke(
    { "messages": [HumanMessage(content="My favourite colour is green")]},
    {"configurable": {"thread_id": "1"}}
)

# %%
from pprint import pprint

pprint(response)

# %%
response = agent.invoke(
    { 
        "messages": [HumanMessage(content="Hello, how are you?")],
        "favourite_colour": "green"
    },
    {"configurable": {"thread_id": "10"}}
)

pprint(response)

# ## Read state

# %%
@tool
def read_favourite_colour(runtime: ToolRuntime) -> str:
    """Read the favourite colour of the user from the state."""
    try:
        return runtime.state["favourite_colour"]
    except KeyError:
        return "No favourite colour found in state"

agent = create_agent(
    "gpt-5-nano",
    tools=[update_favourite_colour, read_favourite_colour],
    checkpointer=InMemorySaver(),
    state_schema=CustomState
)

# %%
response = agent.invoke(
    { "messages": [HumanMessage(content="My favourite colour is green")]},
    {"configurable": {"thread_id": "1"}}
)

pprint(response)

# %%
response = agent.invoke(
    { "messages": [HumanMessage(content="What's my favourite colour?")]},
    {"configurable": {"thread_id": "1"}}
)

pprint(response)

# %%

