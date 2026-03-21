# %%
import asyncio
from pprint import pprint

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from tavily import TavilyClient

load_dotenv()

# MCP Host can also be an Agent or Chat Application. MCP Client resides in MCP Host. MCP Client connect to MCP Server
# LangChain or LangGraph is a Vendor-Agnostic Framework. A vendor-agnostic framework is a design approach that ensures systems, software, or hardware operate independently of any specific vendor, preventing vendor lock-in and promoting interoperability.
# 3rd party MCP Servers are availble at mcp.so
# REFERENCES: To dig deeper into MCP, check out https://learn.deeplearning.ai/courses/mcp-build-rich-context-ai-apps-with-anthropic/information

# %%

load_dotenv()

tavily_client = TavilyClient()

client = MultiServerMCPClient(
    {
        "local_server": {
            "transport": "stdio",
            "command": "python",
            "args": ["2_1_mcp_server.py"],
        }
    }
)


# %%
async def main():
    tools = await client.get_tools()
    resources = await client.get_resources("local_server")
    prompt = await client.get_prompt("local_server", "prompt")
    prompt = prompt[0].content

    agent = create_agent(model="gpt-5-nano", tools=tools, system_prompt=prompt)

    config = {"configurable": {"thread_id": "1"}}

    response = await agent.ainvoke(
        {
            "messages": [
                HumanMessage(content="Tell me about the langchain-mcp-adapters library")
            ]
        },
        config=config,
    )

    pprint(response)


asyncio.run(main())
