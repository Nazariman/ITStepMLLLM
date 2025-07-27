import dotenv
import os
from langchain_google_genai import GoogleGenerativeAI

# завантажити апі ключи 
dotenv.load_dotenv()

# отримати апі ключ 
api_key = os.getenv('GEMINI_API_KEY')

llm = GoogleGenerativeAI(
    model='gemini-2.0-flash',  # назва моделі
    google_api_key=api_key,    # ваша API
)

response = llm.invoke("Привіт, що таке вітання ")
print(response)