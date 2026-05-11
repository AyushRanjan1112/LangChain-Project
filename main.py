from typing import List

from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
# from tavily import TavilyClient
from langchain_tavily import TavilySearch

class Source(BaseModel):
    """Scheme for a source used by the agent"""

    url:str = Field(description="The URL of the source")

class AgentResponse(BaseModel):
    """Schema for agent response with answer and sources"""

    answer: str = Field(description="The agent's answer to the query")
    sources: List[Source] = Field(default_factory=list,description="List of sources used to generate the answer")

class ToolStrategy(Generic[SchemaT]):
    schema: type[SchemaT]
    tool_message_content: str | None
    handle_errors: Union[
        bool,
        str,
        type[Exception],
        tuple[type[Exception], ...],
        Callable[[Exception], str],
    ]

# tavily = TavilyClient()

# Custom tool
""" 
@tool
def search(query: str) -> str:
    \""" 
    Tool that searches over internet
    Args:
        query: The query to search for
    Returns:
        The search result
     \"""
    print(f"Searching for {query}")
    return tavily.search(query=query) """




llm = ChatOllama(model="gpt-oss:20b-cloud",temperature=0,)
tools = [TavilySearch()]

agent = create_agent(model=llm, tools = tools,response_format=AgentResponse)

# HumanMessage -> AIMessage(1st LLM call deciding about the tool and query) -> ToolMessage(as the tool runs with the query, represent execution of tool) -> AIMessage(LangChain made another LLM call and the final answer is that weather in Tokyo is sunny)

# Final LLM call did not choose tool but it shows answer because it had all the information it needed.

def main():
    print("Hello from langchain-course!")
    result = agent.invoke(
        {
            "messages":HumanMessage(
                content="Search for 5 remote/onsite paid internship listing for an ai engineer using langchain in India and list their details"
                )
        }
    )
    print(result)

if __name__ == "__main__":
    main()
