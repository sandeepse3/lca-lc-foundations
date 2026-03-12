# %%
from pprint import pprint
from typing import Any, Dict

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langchain.tools import tool
from tavily import TavilyClient

load_dotenv()

tavily_client = TavilyClient()


@tool
def web_search(query: str) -> Dict[str, Any]:
    """Search the web for information"""
    return tavily_client.search(query)


# Optional quick local test:
# result = web_search.invoke("Who is the current mayor of San Francisco?")
# print(result)

agent = create_agent(
    model="gpt-5-nano",
    tools=[web_search],
)

question = HumanMessage(content="Who is the current mayor of San Francisco?")
response = agent.invoke({"messages": [question]})

print(response["messages"][-1].content)
pprint(response["messages"])
# %%
