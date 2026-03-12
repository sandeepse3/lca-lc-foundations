import asyncio
import sys
from pprint import pprint

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langchain_mcp_adapters.client import MultiServerMCPClient

load_dotenv()

# MCP Host can also be an Agent or Chat Application. MCP Client resides in MCP Host. MCP Client connect to MCP Server
# LangChain or LangGraph is a Vendor-Agnostic Framework. A vendor-agnostic framework is a design approach that ensures systems, software, or hardware operate independently of any specific vendor, preventing vendor lock-in and promoting interoperability.
# 3rd party MCP Servers are availble at mcp.so
# REFERENCES: To dig deeper into MCP, check out https://learn.deeplearning.ai/courses/mcp-build-rich-context-ai-apps-with-anthropic/information


# # Fix for Windows issues in Jupyter notebooks
# if sys.platform == "win32":
#     # 1. Use ProactorEventLoop for subprocess support
#     if not isinstance(asyncio.get_event_loop_policy(), asyncio.WindowsProactorEventLoopPolicy):
#         asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

#     # 2. Redirect stderr to avoid fileno() error when launching MCP servers
#     if "ipykernel" in sys.modules:
#         sys.stderr = sys.__stderr__

# ## Local MCP server

client = MultiServerMCPClient(
    {
        "local_server": {
            "transport": "stdio",
            "command": "python",
            "args": ["resources/2.1_mcp_server.py"],
        }
    }
)


# get tools
async def local_mcp_workflow():
    tools = await client.get_tools()

    # get resources
    resources = await client.get_resources("local_server")

    # get prompts
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


# ## Online MCP
async def online_mcp_workflow():
    client = MultiServerMCPClient(
        {
            "time": {
                "transport": "stdio",
                "command": "uvx",
                "args": ["mcp-server-time", "--local-timezone=America/New_York"],
            }
        }
    )

    tools = await client.get_tools()

    agent = create_agent(
        model="gpt-5-nano",
        tools=tools,
    )

    question = HumanMessage(content="What time is it?")

    response = await agent.ainvoke({"messages": [question]})

    pprint(response)


# Entry point for running workflows
if __name__ == "__main__":
    asyncio.run(local_mcp_workflow())
    asyncio.run(online_mcp_workflow())
