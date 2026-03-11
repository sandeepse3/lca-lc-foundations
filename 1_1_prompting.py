# %%
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage
from pydantic import BaseModel
from pprint import pprint
# temperature
# max tokens
# timeout
# max retries
# https://docs.langchain.com/oss/python/integrations/chat
model = init_chat_model(model="gpt-5-nano")
# %%
system_prompt1 = """

You are a science fiction writer, create a space capital city at the users request.

Please keep to the below structure.

Name: The name of the capital city

Location: Where it is based

Vibe: 2-3 words to describe its vibe

Economy: Main industries

"""
system_prompt = "You are a science fiction writer, create a capital city at the users request."
class CapitalInfo(BaseModel):
    name: str
    location: str
    vibe: str
    economy: str
model = init_chat_model(model="gpt-5-nano", temperature=1.0)

question = HumanMessage(content="What is the capital of The Moon?")
scifi_agent = create_agent(
    model=model,
    system_prompt=system_prompt,
    response_format=CapitalInfo
)

response = scifi_agent.invoke({"messages": [question]})
pprint(response)
# %%
pprint(response['messages'][-1].content)
