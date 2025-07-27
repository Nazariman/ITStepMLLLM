import dotenv
import os
from langchain_google_genai import GoogleGenerativeAI

# Зчитування API ключа з .env
dotenv.load_dotenv()
api_key = os.getenv('GEMINI_API_KEY')

# Ініціалізація моделі
llm = GoogleGenerativeAI(
    model='gemini-2.0-flash',
    google_api_key=api_key,
)

# Зчитування інструкції з файлу
with open("return_policy.txt", encoding="utf-8") as f:
    instruction = f.read().strip()

# Початковий чат-історія
chat_history = f"Instruction: {instruction}\n"

print("Поставте запитання щодо повернення товару (натисніть Enter для завершення):")

while True:
    user_input = input("Human: ").strip()
    if not user_input:
        break

    chat_history += f"Human: {user_input}\n"

    # Виклик LLM для генерації відповіді
    try:
        ai_response = llm.invoke(chat_history)
    except Exception as e:
        ai_response = "Виникла помилка під час виклику моделі: " + str(e)

    # Додавання відповіді до історії
    chat_history += f"AI: {ai_response}\n"

    # Вивід відповіді
    print("AI:", ai_response)

print("\nДякуємо за звернення!")
