from typing import TypedDict,Sequence
from typing import Annotated
from langchain_core.messages import BaseMessage,SystemMessage,HumanMessage
from langgraph.graph.message import add_messages


class MessageState(TypedDict):
    messages: Annotated[list[BaseMessage],add_messages]
    account_balance: str
    statement_summary: str