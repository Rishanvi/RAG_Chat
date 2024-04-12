"""stepback module"""

__all__=['s_display']

import os
import chromadb

from langchain_community.llms import HuggingFaceEndpoint
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.load import dumps, loads
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda


os.environ["HUGGINGFACEHUB_API_TOKEN"]="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"

model_id="mistralai/Mistral-7B-Instruct-v0.2"

embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

db= Chroma(persist_directory="knowledge_base", embedding_function=embeddings)

retriever=db.as_retriever(search_kwargs={"k":5})

model=HuggingFaceEndpoint(repo_id=model_id,max_length=128,temperature=0.5)


examples = [
    {
        "input": "Could the members of The Police perform lawful arrests?",
        "output": "what can the members of The Police do?",
    },
    {
        "input": "Jan Sindel’s was born in what country?",
        "output": "what is Jan Sindel’s personal history?",
    },
]
# We now transform these to example messages
example_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "{input}"),
        ("ai", "{output}"),
    ]
)
few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
)
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an expert at world knowledge. Your task is to step back and paraphrase a question to a more generic step-back question, which is easier to answer. Here are a few examples:""",
        ),
        # Few shot examples
        few_shot_prompt,
        # New question
        ("user", "{question}"),
    ]
)
generate_queries_step_back = prompt | model| StrOutputParser()

# Response prompt 
response_prompt_template = """You are an expert of world knowledge. I am going to ask you a question. Your response should be comprehensive and not contradicted with the following context if they are relevant. Otherwise, ignore them if they are not relevant.

# {normal_context}
# {step_back_context}

# Original Question: {question}
# Answer:"""
response_prompt = ChatPromptTemplate.from_template(response_prompt_template)

chain = (
    {
        # Retrieve context using the normal question
        "normal_context": RunnableLambda(lambda x: x["question"]) | retriever,
        # Retrieve context using the step-back question
        "step_back_context": generate_queries_step_back | retriever,
        # Pass on the question
        "question": lambda x: x["question"],
    }
    | response_prompt
    | model
    | StrOutputParser()
)
















def s_display(question):
   result=chain.invoke({"question": question})
   return result

if(__name__=='__main__'):
    question = input("say something.....")
    result=s_display(question)
    print(result)

