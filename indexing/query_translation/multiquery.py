"""multiquery module"""

__all__=['m_display']

import os
import chromadb

from langchain_community.llms import HuggingFaceEndpoint
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.load import dumps, loads
from operator import itemgetter



embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

db= Chroma(persist_directory="knowledge_base", embedding_function=embeddings)

retriver=db.as_retriever(search_kwargs={"k":5})

os.environ["HUGGINGFACEHUB_API_TOKEN"]="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"
model_id="mistralai/Mistral-7B-Instruct-v0.2"

model=HuggingFaceEndpoint(repo_id=model_id,max_length=128,temperature=0.5)

question_template = """You are an AI language model assistant. Your task is to generate five 
different versions of the given user question to retrieve relevant documents from a vector 
database. By generating multiple perspectives on the user question, your goal is to help
the user overcome some of the limitations of the distance-based similarity search. 
Provide these alternative questions separated by newlines. Original question: {question}"""

promptforquetion=PromptTemplate.from_template(template=question_template)


question_generation_chain= (
    promptforquetion
    | model
    | StrOutputParser() 
    | (lambda x: x.split("\n"))
)


def get_docs(documents: list[list]):
    flattened_docs = [dumps(doc) for sublist in documents for doc in sublist]
    unique_docs = list(set(flattened_docs))
    print("hai")
    return [loads(doc) for doc in unique_docs]



retrieval_chain = question_generation_chain | retriver.map() | get_docs















template=""" <s> [INST]
    since you are a chatbot if you  have  answer for user quetion based on the given context answer properly otherwise make friendly conversation:
    {context}

    user_Question: {question}
    [INST]"""


prompt=PromptTemplate.from_template(template=template)


ragchain={"question":itemgetter("question"),"context":retrieval_chain} | prompt | model |  StrOutputParser()

def m_display(query):

    return ragchain.invoke({"question":query})

if(__name__=='__main__'):
    result= m_display("tell me about cook book")
    print(result)
    