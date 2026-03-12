# %%
from tavily import TavilyClient
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langchain.tools import tool
from dotenv import load_dotenv
from typing import Dict, Any
from pprint import pprint
from langchain.chat_models import init_chat_model

load_dotenv()

tavily_client = TavilyClient()

@tool
def web_search(query: str) -> Dict[str, Any]:
    "Search the web for information"
    return tavily_client.search(query)

model = init_chat_model("gpt-5-nano")
agent = create_agent(
    model=model,
    tools=[web_search],
)

response = agent.invoke({"messages": [HumanMessage(content="What's is about naval mines in present Iran War?")]})
pprint(response["messages"][-1].content)

# %%
pprint(response)