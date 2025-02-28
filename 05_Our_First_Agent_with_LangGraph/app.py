import os
import chainlit as cl
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.tools.arxiv.tool import ArxivQueryRun
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langchain.tools import tool
import wikipediaapi

# 🔹 1️⃣ Load environment variables
load_dotenv()

# 🔹 2️⃣ Retrieve API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT", "AIE5-LangGraph")

# 🔹 3️⃣ Define Tools
tavily_tool = TavilySearchResults(max_results=5)  # Web search
arxiv_tool = ArxivQueryRun()  # Research paper search

# Wikipedia tool
@tool
def wikipedia_search(query: str) -> str:
    """
    Searches Wikipedia for the given query and returns a summary.
    """
    wiki = wikipediaapi.Wikipedia("en")
    page = wiki.page(query)

    if page.exists():
        return page.summary[:500]  # Return first 500 characters
    else:
        return "No Wikipedia page found for this topic."

tool_belt = [tavily_tool, arxiv_tool, wikipedia_search]  # Added Wikipedia

# 🔹 4️⃣ Define the LLM (GPT-4o) and Bind Tools
model = ChatOpenAI(model="gpt-4o", temperature=0)
model = model.bind_tools(tool_belt)  # Enable tool usage

# 🔹 5️⃣ Define the Agent State
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

# 🔹 6️⃣ Define AI Call Function
def call_model(state):
    messages = state["messages"]
    response = model.invoke(messages)
    return {"messages": messages + [response]}  # Append AI response

# 🔹 7️⃣ Define Tool Node (Executes External Tools)
tool_node = ToolNode(tool_belt)

# 🔹 8️⃣ Define State Graph
graph = StateGraph(AgentState)

graph.add_node("agent", call_model)
graph.add_node("action", tool_node)
graph.add_edge("action", "agent")  # Return to agent after tool execution

# 🔹 9️⃣ Define Decision-Making Logic
def should_continue(state):
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "action"
    return END  # End conversation if no tool is needed

graph.add_conditional_edges("agent", should_continue)
graph.set_entry_point("agent")  # Start from 'agent'

# 🔹 🔟 Compile Graph for Execution
compiled_agent = graph.compile()

# 🔹 🔟 Chainlit Bot: Handle User Messages
@cl.on_message
async def on_message(message: cl.Message):
    inputs = {"messages": [HumanMessage(content=message.content)]}

    async for chunk in compiled_agent.astream(inputs, stream_mode="updates"):
        for node, values in chunk.items():
            response = values["messages"][-1]  # Get the last response

            # 🔹 Check if tool calls were made
            if hasattr(response, "tool_calls") and response.tool_calls:
                tool_name = response.tool_calls[0]["name"]
                query = response.tool_calls[0]["args"]["query"]
                formatted_response = f"🤖 GPT-4o requested `{tool_name}` for: **{query}**"
            else:
                formatted_response = response.content  # Use normal AI response

            await cl.Message(content=f"({node}) {formatted_response}").send()
