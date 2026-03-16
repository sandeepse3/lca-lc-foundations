# %%
from pprint import pprint

from dotenv import load_dotenv

load_dotenv()

# MCP Host can also be an Agent or Chat Application. MCP Client resides in MCP Host. MCP Client connect to MCP Server
# LangChain or LangGraph is a Vendor-Agnostic Framework. A vendor-agnostic framework is a design approach that ensures systems, software, or hardware operate independently of any specific vendor, preventing vendor lock-in and promoting interoperability.
# 3rd party MCP Servers are availble at mcp.so
# REFERENCES: To dig deeper into MCP, check out https://learn.deeplearning.ai/courses/mcp-build-rich-context-ai-apps-with-anthropic/information

import asyncio
import sys

# %%
from langchain_mcp_adapters.client import MultiServerMCPClient

# ## Online MCP
client = MultiServerMCPClient(
    {
        "time": {
            "transport": "stdio",
            "command": "uvx",
            "args": ["mcp-server-time", "--local-timezone=Asia/Kolkata"],
        }
    }
)

# %%
# get tools
tools = await client.get_tools()

# %%
from langchain.agents import create_agent

agent = create_agent(model="gpt-5-nano", tools=tools)

pprint(tools)
# %%
agent = create_agent(
    model="gpt-5-nano",
    tools=tools,
)

# %%
from langchain.messages import HumanMessage

question = HumanMessage(content="What time is it?")

response = await agent.ainvoke({"messages": [question]})

pprint(response)

# %%
