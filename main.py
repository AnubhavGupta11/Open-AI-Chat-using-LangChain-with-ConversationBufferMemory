#integrate our code with OpenAI API
import os
from constants import openai_key
from langchain.llms import OpenAI

import streamlit as st

os.environ['OPENAI_API_KEY']=openai_key
#streamlit framework

st.title("LangChain Demo with OpenAI API")
input_text=st.text_input("Search the topic U want")

##
llm=OpenAI(temperature=0.8)

if input_text:
    st.write(llm(input_text))
