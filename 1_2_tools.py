# %%
from langchain.tools import tool
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from pprint import pprint

load_dotenv()
# %%
# @tool
# def square_root(x: float) -> float:
#     """Calculate the square root of a number"""
#     return x**0.5


# @tool("square_root")
# def tool1(x: float) -> float:
#     """Calculate the square root of a number"""
#     return x**0.5

@tool("square_root", description="Calculate the square root of a number")
def tool1(x: float) -> float:
    return x**0.5


tool1.invoke({"x": 467})
# %%
agent = create_agent(
    model="gpt-5-nano",
    tools=[tool1],
    system_prompt="You are an arithmetic wizard. Use your tools to calculate the square root and square of any number.",
)


question = HumanMessage(content="What is the square root of 467?")

response = agent.invoke({"messages": [question]})

print(response["messages"][-1].content)
# %%
pprint(response["messages"])
# %%
print(response["messages"][1].tool_calls)

# %%
