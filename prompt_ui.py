from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import streamlit as st
import os

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-72B-Instruct",
    task="text-generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN"))


model = ChatHuggingFace(
    llm = llm
)

st.header('Research Tool')

user_input = st.text_input('Enter Your Prompt')

if st.button('Summarize'):
    result = model.invoke(user_input)
    st.write(result.content)