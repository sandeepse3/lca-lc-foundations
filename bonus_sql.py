# %%
from dotenv import load_dotenv

load_dotenv()

# %%
from langchain_community.utilities import SQLDatabase

db = SQLDatabase.from_uri("sqlite:///resources/Chinook.db")

# %%
from langchain.tools import tool

@tool
def sql_query(query: str) -> str:

    """Obtain information from the database using SQL queries"""

    try:
        return db.run(query)
    except Exception as e:
        return f"Error: {e}"

sql_query.invoke("SELECT * FROM Artist LIMIT 10")

# %%
from langchain.agents import create_agent

agent = create_agent(
    model="gpt-5-nano",
    tools=[sql_query]
)

# %%
from langchain.messages import HumanMessage

question = HumanMessage(content="Who is the most popular artist beginning with 'S' in this database?")

response = agent.invoke(
    {"messages": [question]}
)

# %%
from pprint import pprint

pprint(response['messages'])

# %%
print(response["messages"][-3].tool_calls[0]['args']['query'])

# %%

