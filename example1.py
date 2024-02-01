#integrate our code with OpenAI API
import os
from constants import openai_key
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain import PromptTemplate
from langchain.chains import SimpleSequentialChain, SequentialChain

import streamlit as st

os.environ['OPENAI_API_KEY']=openai_key
#streamlit framework

st.title("LangChain Demo with OpenAI API")
input_text=st.text_input("Search the topic U want")

first_input_prompt=PromptTemplate(
    input_variables=['name'],
    template="Tell me about celebrity {name}"
)
##
llm=OpenAI(model='gpt-3.5-turbo-instruct',temperature=0.8)
chain1=LLMChain(llm=llm,prompt=first_input_prompt,verbose=True,output_key='person')

second_input_prompt=PromptTemplate(
    input_variables=['person'],
    template="when was {person} born"
)
chain2=LLMChain(llm=llm,prompt=second_input_prompt,verbose=True,output_key='dob')

parent_chain=SequentialChain(chains=[chain1,chain2],input_variables=['name'],output_variables=['person','dob'],verbose=True)

if input_text:
    st.write(parent_chain({'name':input_text}))
