# config.py
import os
from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv

load_dotenv()

def get_chat_model():
    """
    Returns the configured AzureChatOpenAI instance.
    Assumes env vars: AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_VERSION
    """
    # In a real production env, ensure these are set. 
    # For this example, we assume they are injected or in .env
    return AzureChatOpenAI(
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
        temperature=0,
        streaming=True
    )
