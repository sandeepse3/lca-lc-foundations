# %%
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import AIMessage, HumanMessage

# temperature
# max tokens
# timeout
# max retries
# https://docs.langchain.com/oss/python/integrations/chat
model = init_chat_model(model="gpt-5-nano")
# %%
response = model.invoke("What's the capital of Moon?")
print(response)
# %%
print(type(response))
print(response.response_metadata)
# %%
agent = create_agent(model=model)

response = agent.invoke(
    {"messages": [HumanMessage(content="What's the capital of the Moon?")]}
)
print(response)
# %%
print(response["messages"][-1].content)
# %%
response = agent.invoke(
    {
        "messages": [
            HumanMessage(content="What's the capital of the Moon?"),
            AIMessage(content="The capital of the Moon is Luna City."),
            HumanMessage(content="Interesting, tell me more about Luna City"),
        ]
    }
)
print(response)
# %%
print(response["messages"][-1].content)
# %%
for token, metadata in agent.stream(
    {
        "messages": [
            HumanMessage(content="Tell me all about Luna City, the capital of the Moon")
        ]
    },
    stream_mode="messages",
):
    # token is a message chunk with token content
    # metadata contains which node produced the token

    if token.content:  # Check if there's actual content
        print(token.content, end="", flush=True)  # Print token
# %%
