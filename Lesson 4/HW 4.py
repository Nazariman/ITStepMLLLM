from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
import json
import dotenv
import os

# Завантаження API ключа з .env
dotenv.load_dotenv()
api_key = os.getenv('GEMINI_API_KEY')

# Меню піцерії
menu = {
    "Маргарита": {"Мала": 120, "Велика": 180},
    "Пепероні": {"Мала": 140, "Велика": 200},
    "4 сири": {"Мала": 160, "Велика": 220},
    "Гавайська": {"Мала": 150, "Велика": 210},
    "Вегетаріанська": {"Мала": 130, "Велика": 190}
}

# Словник для зберігання замовлення
order = {}

# Створення LLM
llm = ChatGoogleGenerativeAI(
    model='gemini-2.0-flash',
    google_api_key=api_key
)

# Prompt для поведінки бота
template = """
Ти ввічливий чат-бот піцерії. Твої завдання:
1. Показати меню, якщо користувач попросить.
2. Прийняти замовлення, запитавши назву піци, розмір та кількість.
3. Дозволити змінювати замовлення.
4. Підтвердити замовлення, підрахувавши суму.

Меню (назва - ціна Мала/Велика):
{menu_json}

Користувач: {user_input}
Відповідай українською.
"""

prompt = PromptTemplate(
    input_variables=["menu_json", "user_input"],
    template=template
)

def chat_with_bot(user_input):
    formatted_prompt = prompt.format(
        menu_json=json.dumps(menu, ensure_ascii=False),
        user_input=user_input
    )

    response = llm([
        SystemMessage(content="Ти працюєш як бот замовлень піци."),
        HumanMessage(content=formatted_prompt)
    ])
    return response.content

while True:
    user_input = input("Ви: ")
    if user_input.lower() in ["вихід", "exit"]:
        print("Бот: Дякуємо, гарного дня!")
        break
    answer = chat_with_bot(user_input)
    print("Бот:", answer)
