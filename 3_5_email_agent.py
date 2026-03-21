from dataclasses import dataclass
from typing import Callable

from dotenv import load_dotenv
from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import (
    HumanInTheLoopMiddleware,
    ModelRequest,
    ModelResponse,
    dynamic_prompt,
    wrap_model_call,
)
from langchain.messages import HumanMessage, ToolMessage
from langchain.tools import tool, ToolRuntime
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from pprint import pprint

load_dotenv()


# email agent
# - authenticates user
#     - only then are they allowed into the "inbox"
#     - dynamic tools and prompt on the condition of there being an email and password in state that match hardcoded
# - checks "inbox"
#     - email in tool
# - sends emails
#     - human in the loop


@dataclass
class EmailContext:
    email_address: str = "julie@example.com"
    password: str = "password123"


class AuthenticatedState(AgentState):
    authenticated: bool


@tool
def check_inbox() -> str:
    """Check the inbox for recent emails"""
    return """
    Hi Julie, 
    I'm going to be in town next week and was wondering if we could grab a coffee?
    - best, Jane (jane@example.com)
    """


@tool
def send_email(to: str, subject: str, body: str) -> str:
    """Send an response email"""
    return f"Email sent to {to} with subject {subject} and body {body}"


@tool
def authenticate(email: str, password: str, runtime: ToolRuntime) -> Command:
    """Authenticate the user with the given email and password"""
    if email == runtime.context.email_address and password == runtime.context.password:
        return Command(
            update={
                "authenticated": True,
                "messages": [
                    ToolMessage("Successfully authenticated", tool_call_id=runtime.tool_call_id)
                ],
            }
        )
    else:
        return Command(
            update={
                "authenticated": False,
                "messages": [
                    ToolMessage("Authentication failed", tool_call_id=runtime.tool_call_id)
                ],
            }
        )


authenticated_prompt = "You are a helpful assistant that can check the inbox and send emails."
unauthenticated_prompt = "You are a helpful assistant that can authenticate users."


@wrap_model_call
def dynamic_tool_call(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:

    """Allow read inbox and send email tools only if user provides correct email and password"""

    authenticated = request.state.get("authenticated")

    if authenticated:
        tools = [check_inbox, send_email]
    else:
        tools = [authenticate]

    request = request.override(tools=tools)
    return handler(request)


@dynamic_prompt
def dynamic_prompt(request: ModelRequest) -> str:
    """Generate system prompt based on authentication status"""
    authenticated = request.state.get("authenticated")

    if authenticated:
        return authenticated_prompt
    else:
        return unauthenticated_prompt


agent = create_agent(
    "gpt-5-nano",
    tools=[authenticate, check_inbox, send_email],
    checkpointer=InMemorySaver(),
    state_schema=AuthenticatedState,
    context_schema=EmailContext,
    middleware=[
        dynamic_tool_call,
        dynamic_prompt,
        HumanInTheLoopMiddleware(
            interrupt_on={
                "authenticate": False,
                "check_inbox": False,
                "send_email": True,
            }
        ),
    ],
)


config = {"configurable": {"thread_id": "1"}}

response = agent.invoke(
    {"messages": [HumanMessage(content="draft 1")]},
    context=EmailContext(),
    config=config,
)

print(response["messages"][-1].content)

# %%
print(response["__interrupt__"][0].value["action_requests"][0]["args"]["body"])

# %%
response = agent.invoke(
    Command(
        resume={"decisions": [{"type": "approve"}]}  # or "reject"
    ),
    config=config,  # Same thread ID to resume the paused conversation
)

print(response["messages"][-1].content)

# %%
pprint(response)

# %%
