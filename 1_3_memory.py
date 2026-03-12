# %%
from pprint import pprint
from dotenv import load_dotenv
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents import create_agent
from langchain.messages import HumanMessage

load_dotenv()


agent = create_agent(
    model="gpt-5-nano",
    checkpointer=InMemorySaver()
)
config = {"configurable": {"thread_id": "1"}}
question = HumanMessage(content="This is Sunny and I like black color")
response = agent.invoke({"messages": [question]}, config=config)

print(response["messages"][-1].content)
# %%
question1 = HumanMessage(content="What is my name and what color do I like?")
response = agent.invoke({"messages": [question1]}, config=config)
pprint(response["messages"])
# %%
