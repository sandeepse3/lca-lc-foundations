# %%
from pprint import pprint

from dotenv import load_dotenv
from langchain.agents import AgentState, create_agent
from langchain.messages import HumanMessage, ToolMessage
from langchain.tools import ToolRuntime, tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

load_dotenv()


# %%
class CustomState(AgentState):
    favourite_colour: str


# ## Write to state
# It just updates the state using a update tool. And ad


# %%
@tool
def update_favourite_colour(favourite_colour: str, runtime: ToolRuntime) -> Command:
    """Update the favourite colour of the user in the state once they've revealed it."""
    return Command(
        update={
            "favourite_colour": favourite_colour,
            "messages": [
                ToolMessage(
                    "Successfully updated favourite colour",
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        }
    )


# %%
agent = create_agent(
    "gpt-5-nano",
    tools=[update_favourite_colour],
    checkpointer=InMemorySaver(),
    state_schema=CustomState,
)


# %%
response = agent.invoke(
    {"messages": [HumanMessage(content="My favourite colour is red")]},
    {"configurable": {"thread_id": "1"}},
)


# %%
pprint(response)
