import streamlit as st
import langchain
import langchain_huggingface
from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
import dotenv
from dotenv import load_dotenv
import os

#load env variables
load_dotenv()

# deepSeek API key
os.environ["HF_TOKEN"] = os.getenv("hf")

st.title("AI Conversational Chatbot using Hugging Face Endpoint")

# acessing the deepseek model
model = HuggingFaceEndpoint(repo_id = "deepseek-ai/DeepSeek-V3.2",temperature = 0.2)
deepseek = ChatHuggingFace(llm = model)

# Initialize memory
if "conver" not in st.session_state:
    st.session_state["conver"] = []
    st.session_state["memory"] = []
    st.session_state["memory"].append(("system","""You are an intelligent, friendly, and professional AI assistant.
                                    Your purpose is to help users with general questions, learning, problem-solving, and project-related guidance.
                                    Provide clear, accurate, and easy-to-understand explanations.
                                    Adapt your responses to the user's level of knowledge.
                                    When handling technical or project-related queries, give structured explanations, clean examples, and best practices.
                                    Identify and correct mistakes politely and explain the reason behind them.
                                    If a question is unclear, ask for clarification.
                                    If you are unsure about an answer, respond honestly without guessing.
                                    Maintain a supportive, respectful, and encouraging tone at all times.
                                    Focus on being helpful, reliable, and efficient."""))

user_data = st.chat_input("user_message")

if user_data:
    st.session_state["memory"].append(("human",user_data))

    with st.chat_message(("human")):
        st.write(user_data)

    output = deepseek.invoke(st.session_state["memory"])

    st.session_state["memory"].append({"ai",output.content})
    st.session_state["conver"].append({"role":"human","data":user_data})
    st.session_state["conver"].append({"role":"ai","data":output.content})

    if user_data == "bye":
        st.session_state["conver"] = []
        st.session_state["memory"] = []

for y in st.session_state["conver"]:
    with st.chat_message(y["role"]):
        st.write(y["data"])


