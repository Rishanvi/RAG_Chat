#````````````````````````````````````````````#
#````````````WARNING  INSTALL NEEDED PACKAGES```````````````#
#````````````````````````````````````````````#
#   **use upgraded packages **

# pip install streamlit
# pip install beautifulsoup4
# pip install requests
# pip install sentence-transformers
# pip install langchain
# pip install chromadb
# pip install huggingface-hub



#````````````````````````````````````````````#
#````````````WARNING  INSTALL NEEDED PACKAGES```````````````#
#````````````````````````````````````````````#


#````````````````````````````````````````````#
#````````````IMPORTING PACKAGES```````````````#
#````````````````````````````````````````````#

import streamlit as st    
import os                 
import chromadb 
import bs4               
import requests

from langchain_community.llms.huggingface_endpoint import HuggingFaceEndpoint
from langchain.chains import LLMChain
from langchain_community.document_loaders import TextLoader , WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain.chains import RetrievalQA,ConversationChain
from langchain.memory import ConversationBufferMemory


#````````````````````````````````````````````#
#````````````IMPORTING PACKAGES```````````````#
#````````````````````````````````````````````#


#````````````````````````````````````````````#
#````````````HUGGINGFACE ACCESS TOKEN```````````````#
#````````````````````````````````````````````#

#os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"
#hugging face tokens are varied for different user  

os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"
#HUGGINGFACEHUB_API_TOKEN = "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"


#````````````````````````````````````````````#
#````````````HUGGINGFACE ACCESS TOKEN```````````````#
#````````````````````````````````````````````#




#````````````````````````````````````````````#
#````````````EMBEDDING MODEL```````````````#
#````````````````````````````````````````````#

embeddings= HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


#````````````````````````````````````````````#
#````````````EMBEDDING MODEL```````````````#
#````````````````````````````````````````````#


#````````````````````````````````````````````#
#````````````SET RETRIVER```````````````#
#````````````````````````````````````````````#

def data_base():

    
    # loader=WebBaseLoader(url)
    # text=loader.load()
    # text_splitter=RecursiveCharacterTextSplitter(chunk_size=300,chunk_overlap=10)
    # docs=text_splitter.split_documents(text)


    vector_store=Chroma(persist_directory="./knowledge_base", embedding_function=embeddings)
    retriever = vector_store.as_retriever(search_type="similarity",search_kwargs={"k":5})
    return retriever

#````````````````````````````````````````````#
#````````````SET RETRIVER```````````````#
#````````````````````````````````````````````#




#````````````````````````````````````````````#
#````````````SET RAG CHAIN```````````````#
#````````````````````````````````````````````#

def get_response(retrive,quetion):

    template = """ <s> [INST]
    since you are a chatbot if you  have  answer for user quetion based on the given context answer properly otherwise make friendly conversation:
    {context}

    user_Question: {question}
    [INST]"""
    prompt = PromptTemplate.from_template(template)

    memory=ConversationBufferMemory(ai_prifix="AI Assistant")

    repo_id = "mistralai/Mistral-7B-Instruct-v0.2"
    llm = HuggingFaceEndpoint(
        repo_id=repo_id, max_length=128, temperature=0.5)


    # llm_chain = LLMChain(prompt=prompt, llm=llm)


    # chain = (
    #     {"context": retrive, "question": RunnablePassthrough()}
    #     | prompt
    #     | llm
    #     | StrOutputParser()
    # )
    
    chain=RetrievalQA.from_chain_type(llm=llm,retriever=retrive,chain_type='stuff', chain_type_kwargs={"prompt":prompt},return_source_documents=True)

    result=chain({"query":quetion})
    return result['result']

    # conversation=ConversationChain(llm=llm,memory=memory,verbose=True)
    # result=conversation.predict(input=quetion)

    #return conversation.memory.buffer
 #````````````````````````````````````````````#
#````````````SET RAG CHAIN```````````````#
#````````````````````````````````````````````#


#````````````````````````````````````````````#
#````````````STREAMLIT SETUP```````````````#
#````````````````````````````````````````````#

# if 'store_message' not in st.session_state:
#     st.session_state.store_message=[]
    

# side_bar=st.sidebar

# with side_bar:
#     text_input=side_bar.text_input("enter something")
# if(text_input!=""):
#     st.title(":red[#]:blue[#] C:blue[H]A:red[T] A:blue[SS]I:red[S]T:green[A]N:red[T] :blue[#]:red[#]")
    
#     container=st.container(height=400)
#     message=st.chat_input("say somthing.....")
#     demo_chat=container.chat_message(name="user",avatar="💀")
#     with demo_chat:
#         demo_chat.write("I' am :blue[AI] :red[BOT] # How Can I😭 U")
#     if(message):
#         answer=get_response(data_base(text_input),message)
#         st.session_state.store_message.append({"user":message,"ass":answer})
#         side_bar.write(st.session_state.store_message)
        
#         for chat in st.session_state.store_message:
#             user=container.chat_message(name="user",avatar="😉")
#             with user:
#                 user.write(chat['user'])
#             ass=container.chat_message(name="assistant",avatar="💀")
#             with ass:
#                 ass.write(chat['ass'])    
            

# else:
#     st.info("enter any website url first")

side_bar=st.sidebar


if 'store_message' not in st.session_state:
    st.session_state.store_message=[]

st.title(":red[#]:blue[#] C:blue[H]A:red[T] A:blue[SS]I:red[S]T:green[A]N:red[T] :blue[#]:red[#]")

container=st.container(height=400)
message=st.chat_input("say somthing.....")
demo_chat=container.chat_message(name="user",avatar="💀")
with demo_chat:
    demo_chat.write("I' am :blue[AI] :red[BOT] # How Can I😭 U")
    
if(message):
    answer=get_response(data_base(),message)
    st.session_state.store_message.append({"user":message,"ass":answer})
    side_bar.write(st.session_state.store_message)    

    for chat in st.session_state.store_message:
        user=container.chat_message(name="user",avatar="😉")
        with user:
            user.write(chat['user'])
        ass=container.chat_message(name="assistant",avatar="💀")
        with ass:
            ass.write(chat['ass'])    
    

#````````````````````````````````````````````#
#````````````STREAMLIT SETUP```````````````#
#````````````````````````````````````````````#    
