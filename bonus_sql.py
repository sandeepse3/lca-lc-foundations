from dotenv import load_dotenv
from langchain_community.utilities import SQLDatabase
from langchain.tools import tool
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from pprint import pprint


load_dotenv()


db = SQLDatabase.from_uri("sqlite:///resources/Chinook.db")


@tool
def sql_query(query: str) -> str:

    """Obtain information from the database using SQL queries"""

    try:
        return db.run(query)
    except Exception as e:
        return f"Error: {e}"


sql_query.invoke("SELECT * FROM Artist LIMIT 10")


agent = create_agent(
    model="gpt-5-nano",
    tools=[sql_query]
)


question = HumanMessage(content="Who is the most popular artist beginning with 'S' in this database?")

response = agent.invoke(
    {"messages": [question]}
)


pprint(response['messages'])


print(response["messages"][-3].tool_calls[0]['args']['query'])

# %%
