from langchain_groq import ChatGroq
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType
from langchain.document_loaders import WebBaseLoader
import pandas as pd

llm = ChatGroq(
   
)

df=pd.read_csv("output.csv")
agent=create_pandas_dataframe_agent(llm=llm,df=df,verbose=False,allow_dangerous_code=True)
def rag_qa(query: str):
    """
    This function is used when the user wants any information regarding his bank account statement.

    Args:
    query: The question asked by the user.
    """
    result=agent.invoke(query)
    return result['output']


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

    Args:
    query: The question asked by the user.
    """
    user_link=ask_user()
    scraped_data=scraper(user_link)
    print(scraped_data)
    financial_summary=agent.invoke("give me a summary of my past transactions along with where i spent my money")['output']
    print(financial_summary)
    account_balance=agent.invoke("what is my current account balance")['output']
    print(account_balance)
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
    return result.content


