from typing import Annotated
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing import TypedDict
from langchain_groq import ChatGroq
from tools import rag_qa,to_buy
from langgraph.prebuilt import ToolNode,tools_condition
from langgraph.graph import StateGraph,START,END


llm = ChatGroq(
   
)
class MessageState(TypedDict):
    messages: Annotated[list[BaseMessage],add_messages]


llm_with_tools=llm.bind_tools([rag_qa,to_buy])

def tool_calling_llm(state: MessageState):
    return {"messages":[llm_with_tools.invoke(state["messages"])]}


builder=StateGraph(MessageState)
builder.add_node("tool_calling_llm",tool_calling_llm)
builder.add_node("tools",ToolNode([rag_qa,to_buy]))
builder.add_edge(START,"tool_calling_llm")
builder.add_conditional_edges(
    "tool_calling_llm",tools_condition,
)
builder.add_edge("tools",END)
graph=builder.compile()

print(graph.invoke({"messages":"Can i buy something?"}))