from typing import Annotated
from langchain_core.messages import BaseMessage,SystemMessage,HumanMessage,AIMessage
from langgraph.graph.message import add_messages
from typing import TypedDict,Sequence
from langchain_groq import ChatGroq
from tools import rag_qa,to_buy,imp_info,plot
from langgraph.prebuilt import ToolNode,tools_condition
from langgraph.graph import StateGraph,START,END
from langchain.agents import create_tool_calling_agent,AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from langgraph.prebuilt.tool_executor import ToolExecutor
import operator
from langgraph.prebuilt import ToolInvocation
import json
from state import MessageState
from langgraph.checkpoint.memory import MemorySaver

llm = ChatGroq(
   
)

def agent(state:MessageState):
        user_query=state["messages"][-1]
        ai_response2=llm.invoke([SystemMessage(content=f"You are an intelligent general assistant. You will be given a sentence.You have to tell whether the user wants any information regarding his bank account statement or when the user wants to buy something or when the user wants a chart/plot something.If the user wants any information regarding his bank account statement say bank and if the user talks about buying something say buy and if user asks about creating a plot or showing a chart just say plot. The sentence is {user_query}")])
        if "bank" in ai_response2.content.lower():
             return {"messages":[AIMessage(content="rag_node")]}
        elif "buy" in ai_response2.content.lower():
            return {"messages":[AIMessage(content="ask_2_buy")]}
        elif "plot" in ai_response2.content.lower():
            return {"messages":[AIMessage(content="plot_node")]}

def should_continue(state:MessageState):
      message = state["messages"][-1]
      if message.content == "rag_node":
        return "rag_node"
      elif message.content=="ask_2_buy":
        return "ask_2_buy"
      elif message.content=="plot_node":
          return "plot_node"





builder=StateGraph(MessageState)
builder.add_node("imp_info",imp_info)
builder.add_node("agent",agent)
builder.add_node("rag_node",rag_qa)
builder.add_node("ask_2_buy",to_buy)
builder.add_node("plot_node",plot)
builder.add_edge(START,"imp_info")
builder.add_edge("imp_info","agent")
builder.add_conditional_edges("agent",should_continue,["rag_node","ask_2_buy","plot_node"])
builder.add_edge("rag_node",END)
builder.add_edge("ask_2_buy",END)
builder.add_edge("plot_node",END)
memory=MemorySaver()
app=builder.compile(checkpointer=memory)

user_input = input("How can I help you ?:")
thread={"configurable":{"thread_id":"1"}}
for event in app.stream({"messages": [HumanMessage(content=user_input)]},thread,stream_mode="values"):
    event["messages"][-1].pretty_print()