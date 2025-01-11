from langchain_groq import ChatGroq
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType
from langchain.document_loaders import WebBaseLoader
import pandas as pd
from state import MessageState
import pandasai
from pandasai import SmartDataframe
from langchain_core.messages import HumanMessage,AIMessage,SystemMessage


llm = ChatGroq(
    ###
)

df=pd.read_csv("output.csv")

agent=SmartDataframe(df,config={"llm":llm})


def imp_info(state: MessageState):
    financial_summary=agent.chat("Give me a summary of my past transactions with transaction at each place in a paragraph")
    account_balance=df['Balance'].iloc[0]
    return {"account_balance":account_balance,"statement_summary":financial_summary}

def rag_qa(state: MessageState):

    query=state["messages"][-2].content
    summary=state["statement_summary"]
    balance=state["account_balance"]
    instructions="""
    You are an intelligent financial agent
    You are provided with the user's financial summary and account balance.
    Using the information provided, you must answer the user's question in a clear and concise manner.

    User's Question : {query}
    Financial Summary : {financial_summary}
    Account Balance : {account_balance}
    """

    system_message=instructions.format(query=query,financial_summary=summary,account_balance=balance)
    result=llm.invoke([SystemMessage(content=system_message)])
    return  {"messages":[AIMessage(content=result.content)]}





def scraper(link):
    loader = WebBaseLoader(link)
    documents = loader.load()
    result=documents[0].page_content
    return result

def ask_user():
    user_input=input("Enter the link of the thing you want to buy:")
    return user_input

def to_buy(query: str):
    """
    This function is called when the user wants to buy something and he wants to ask whether is it feasible to buy it or not.
    Call this only when the user says he/she wants to buy something.

    Args:
    query: The question asked by the user.
    """
    user_link=ask_user()
    scraped_data=scraper(user_link)
    financial_summary=MessageState['statement_summary']
    account_balance=MessageState['account_balance']
    
    instructions="""
    You are an intelligent financial agent
    You are provided with a description of the product the user wants to buy along with the user's financial summary and account balance.
    You must first decided whether this thing the person wants to buy is worth it or not considering the user's bank account statement and account balance.
    Think properly and then give your answer because you are a reputed financial advisor.As a financial advisor, you must give your honest opinion.
    It is your duty to help the user make a wise decision. Whether the person should buy this product or not. 
    If you refrain from purchasing the product please suggest cheaper alternatives along with their links.

    Product description : {scraped_data}
    Financial Summary : {financial_summary}
    Account Balance : {account_balance}
    """

    system_message=instructions.format(scraped_data=scraped_data,financial_summary=financial_summary,account_balance=account_balance)
    result=llm.invoke(system_message)
    return {"messages":result.content}


