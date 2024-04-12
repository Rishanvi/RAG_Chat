"""decomposer module"""

__all__=['d_display']


import os
import chromadb

from langchain_community.llms import HuggingFaceEndpoint
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain import hub
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

db= Chroma(persist_directory="knowledge_base", embedding_function=embeddings)

retriever=db.as_retriever(search_kwargs={"k":5})

os.environ["HUGGINGFACEHUB_API_TOKEN"]="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"
model_id="mistralai/Mistral-7B-Instruct-v0.2"

model=HuggingFaceEndpoint(repo_id=model_id,max_length=128,temperature=0.5)

prompt_rag_template=""" <s> [INST]
    since you are a chatbot if you  have  answer for user quetion based on the given context answer properly otherwise make friendly conversation:
    {context}

    user_Question: {question}
    [INST]"""

prompt_rag = PromptTemplate.from_template(template=prompt_rag_template)

def retrieve_and_rag(question,prompt_rag,sub_question_generator_chain):
    """RAG on each sub-question"""
    
    # Use our decomposition / 
    sub_questions = sub_question_generator_chain.invoke({"question":question})
    
    # Initialize a list to hold RAG chain results
    rag_results = []
    
    for sub_question in sub_questions:
        
        # Retrieve documents for each sub-question
        retrieved_docs = retriever.get_relevant_documents(sub_question)
        
        # Use retrieved documents and sub-question in RAG chain
        answer = (prompt_rag | model| StrOutputParser()).invoke({"context": retrieved_docs, 
                                                                "question": sub_question})
        rag_results.append(answer)
    
    return rag_results,sub_questions

# Wrap the retrieval and RAG process in a RunnableLambda for integration into a chain

template = """You are a helpful assistant that generates multiple sub-questions related to an input question. \n
The goal is to break down the input into a set of sub-problems / sub-questions that can be answers in isolation. \n
Generate multiple search queries related to: {question} \n
Output (3 queries):"""
prompt_decomposition = ChatPromptTemplate.from_template(template)

generate_queries_decomposition = ( prompt_decomposition | model | StrOutputParser() | (lambda x: x.split("\n")))

def format_qa_pairs(questions, answers):
    """Format Q and A pairs"""
    
    formatted_string = ""
    for i, (question, answer) in enumerate(zip(questions, answers), start=1):
        formatted_string += f"Question {i}: {question}\nAnswer {i}: {answer}\n\n"
    return formatted_string.strip()



# Prompt
template = """Here is a set of Q+A pairs:

{context}

Use these to synthesize an answer to the question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

final_rag_chain = (
    prompt
    | model
    | StrOutputParser()
)















def d_display(question):
   
   answers, questions = retrieve_and_rag(question, prompt_rag, generate_queries_decomposition)
   context = format_qa_pairs(questions, answers)
   result=final_rag_chain.invoke({"context":context,"question":question})
   return result
if(__name__=='__main__'):
    question = input("say something.....")
    result=d_display(question)
    print(result)
