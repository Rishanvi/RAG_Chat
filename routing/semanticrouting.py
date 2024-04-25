import os
import chromadb
from langchain_community.llms import HuggingFaceEndpoint
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate 
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter
from dotenv import load_dotenv

from langsmith import Client
Client()
from langchain.utils.math import cosine_similarity
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

db= Chroma(persist_directory="knowledge_base", embedding_function=embeddings)

retriver=db.as_retriever(search_kwargs={"k":15})

model_id="mistralai/Mistral-7B-Instruct-v0.2"

model=HuggingFaceEndpoint(repo_id=model_id,max_length=128,temperature=0.5)

# Two prompts
physics_template = """You are a very smart physics professor. 
You are great at answering questions about physics in a concise and easy to understand manner. 
When you don't know the answer to a question you admit that you don't know.

Here is a question:
{query}"""

math_template = """You are a very good mathematician. You are great at answering math questions. 
You are so good because you are able to break down hard problems into their component parts, 
answer the component parts, and then put them together to answer the broader question.

Here is a question:
{query}"""

# Embed prompts
prompt_templates = [physics_template, math_template,"parallel universe"]
prompt_embeddings = embeddings.embed_documents(prompt_templates)


# Route question to prompt 
def prompt_router(input):
    # Embed question
    query_embedding = embeddings.embed_query(input["query"])
    # Compute similarity
    similarity = cosine_similarity([query_embedding], prompt_embeddings)[0]
    most_similar = prompt_templates[similarity.argmax()]
    # Chosen prompt 
    print("Using MATH" if most_similar == math_template else "Using PHYSICS")
    return PromptTemplate.from_template(most_similar)

chain = (
    {"query": RunnablePassthrough()}
    | RunnableLambda(prompt_router)
    | model
    | StrOutputParser()
)

print(chain.invoke("What's a black hole"))