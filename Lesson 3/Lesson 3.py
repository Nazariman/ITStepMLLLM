from langchain_google_genai import GoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.output_parsers import ResponseSchema, StructuredOutputParser

import json
import dotenv
import os

# завантажити api ключі з папки .env
dotenv.load_dotenv()

# отримати сам ключ
api_key = os.getenv('GEMINI_API_KEY')

# створення моделі
# Велика мовна модель(llm)

llm = GoogleGenerativeAI(
    model='gemini-2.0-flash',  # назва моделі
    google_api_key=api_key,    # ваша API
)

# Користувач задає питання по книзі
# Ваша задача:
# 1. Дати відповідь на питання
# 2. Порекомендувати схожі книги(на ту саму тему, того ж автора, жанру, ...)

# має назву книги і питання -- хочемо отримати всю інформацію про книгу
# і відповідь на питання

# маючи інформацію про книгу порекомендувати щось схоже


# ------------------------------

# схема для відповідей
schemas = [
    ResponseSchema(name='answer', description='відповідь на питання користувача'),
    ResponseSchema(name='theme', description='головна тема книги'),
    ResponseSchema(name='author', description='автор книги'),
    ResponseSchema(name='genre', description='жанр книги')
]

# створення парсер
parser = StructuredOutputParser.from_response_schemas(schemas)

# отримати інструкція для llm
instructions = parser.get_format_instructions()

#print(instructions)

# створення промпта
prompt = PromptTemplate.from_template(
    """
    Ти асистент онлайн книгарні. Твоя задача давати відповіді
    на питання користувачів. Відповіді мають бути чіткі та інформативні.
    Загальний стиль спілкування ввічливий, іноді можеш використовувати
    неформальний стиль.
    Також ти повинен визначити параметри книжки про яку питає
    користувач(наприклад жанр, автор, тема, ...)
    
    Питання: {question}
    
    Формат відповіді:
    {instructions}
    """,
    partial_variables={"instructions": instructions} # Вказує що instructions завжди береться той що прописаний а не підставляється динамічно
)

# створення ланцюга
chain = prompt | llm | parser

response = chain.invoke({
    "question": "Коли була написана книга 1984",
})

print(response)
print(response['theme'])
print(type(response))

# # збереження у файл
# with open('response.json', 'w', encoding='UTF-8') as file:
#     json.dump(response, file)
#
# # завантаження з файлу
# with open('response.json', 'r', encoding='UTF-8') as file:
#     new_response = json.load(file)
#
# print(new_response)


# рекомендація схожих книг
prompt = PromptTemplate.from_template(
    """
    Ти асистент онлайн книгарні. Твоя задача давати рекомендації книг
    користувачам певно жанру, теми та автора. Запропонуй по 3-5 книг по
    кожному пункту.
    Загальний стиль спілкування ввічливий, іноді можеш використовувати
    неформальний стиль.
    
    Жанр: {genre}
    Автор: {author}
    Тема: {theme}
    
    Відповідь дай у вигляді списку, познач які книги до якого 
    пункту відносяться
    * Книги на схожу тему
    * Книги того ж автора
    * Книги того ж жанру
    """
)

chain_recommendation = prompt | llm

recommendation = chain_recommendation.invoke({
    "genre": response['genre'],
    "author": response['author'],
    "theme": response['theme'],
})

# print(recommendation)

# дістати всі назви книг з рекомендації

schemas = [
    ResponseSchema(name='books', description='список з назвами книг')
]

parser = StructuredOutputParser.from_response_schemas(schemas)
instructions = parser.get_format_instructions()

prompt = PromptTemplate.from_template(
    """
    Твоя задача дістати назви усіх книг з тексту.
    
    Текст: {text}
    
    Формат відповіді:
    {instructions}
    """,
    partial_variables={"instructions": instructions}
)

chain_book_selector = prompt | llm | parser

response = chain_book_selector.invoke({
    "text": recommendation
})

print(response)

for book in response['books']:
    print(book)