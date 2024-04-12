from indexing.query_translation.decomposition import d_display
from langsmith import Client
import os
os.environ["LANGCHAIN_API_KEY"] = "ls__08d1b346dd804c139717ead521d66221"
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"] = "rag_chat model"

Client()





if(__name__=="__main__"):
   
    query=input("say something......")
    result= d_display(query)
    print(result)





