from langchain_core.messages import SystemMessage,HumanMessage,AIMessage
from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from dotenv import load_dotenv
import os

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-72B-Instruct",
    task="text-generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")) 

model = ChatHuggingFace(llm = llm)

messages = [
    SystemMessage(content = 'You are a helpful assistant'),
    HumanMessage(content = 'Tell me about LangChain')
]

result = model.invoke(messages)

print(result.content)