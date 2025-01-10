from typing import Annotated
from langchain_core.messages import BaseMessage,SystemMessage,HumanMessage
from langgraph.graph.message import add_messages
from typing import TypedDict,Sequence
from langchain_groq import ChatGroq
from tools import rag_qa,to_buy,imp_info
from langgraph.prebuilt import ToolNode,tools_condition
from langgraph.graph import StateGraph,START,END
from langchain.agents import create_tool_calling_agent,AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from langgraph.prebuilt.tool_executor import ToolExecutor
import operator
from langgraph.prebuilt import ToolInvocation
import json
from state import MessageState
llm = ChatGroq(
    
)



    


llm_with_tools=llm.bind_tools([rag_qa,to_buy])

def tool_calling_llm(state: MessageState):
    return {"messages":[llm_with_tools.invoke(state["messages"]+[SystemMessage(content="You are a helpful assistant. Answer every question in detail. Give your reason for every answer")])]}


builder=StateGraph(MessageState)
builder.add_node("tool_calling_llm",tool_calling_llm)
builder.add_node("tools",ToolNode([rag_qa,to_buy]))
builder.add_node("imp_info",imp_info)
builder.add_edge(START,"imp_info")
builder.add_edge("imp_info","tool_calling_llm")
builder.add_conditional_edges(
    "tool_calling_llm",tools_condition,
)
builder.add_edge("tools",END)
graph=builder.compile()

print(graph.invoke({"messages": [SystemMessage(content="Where did i spend the least money?")]}))