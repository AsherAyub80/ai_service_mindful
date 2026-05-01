import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
print(f"API Key found: {api_key[:5]}...{api_key[-5:] if api_key else ''}")

client = Groq(api_key=api_key)

try:
    models = client.models.list()
    print("Successfully authenticated with Groq!")
    for model in models.data:
        print(f"- {model.id}")
except Exception as e:
    print(f"Failed to authenticate: {e}")
