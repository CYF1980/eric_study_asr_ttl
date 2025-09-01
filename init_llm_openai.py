# https://python.langchain.com/docs/tutorials/llm_chain/
import getpass
import os
from langchain.chat_models import init_chat_model

# Initialize the OpenAI chat model
def get_openai_model():
    return init_chat_model("gpt-4o-mini", model_provider="openai")
