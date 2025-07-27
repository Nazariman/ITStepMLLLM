from langchain_google_genai import GoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
import dotenv
import os

# Завантаження API ключа
dotenv.load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

llm = GoogleGenerativeAI(
    model='gemini-2.0-flash',
    google_api_key=api_key
)


# Перший ланцюг

# Схема для вправ
schemas_exercises = [
    ResponseSchema(name='goal', description='мета тренування'),
    ResponseSchema(name='exercises', description='список рекомендованих вправ')
]

parser_exercises = StructuredOutputParser.from_response_schemas(schemas_exercises)
instructions_ex = parser_exercises.get_format_instructions()

prompt_exercises = PromptTemplate.from_template(
    """
    Ти — фітнес-асистент. Твоя задача — підібрати список вправ для користувача відповідно до його мети тренування.
    
    Мета: {goal}
    
    Формат відповіді:
    {instructions}
    """,
    partial_variables={"instructions": instructions_ex}
)

chain_exercises = prompt_exercises | llm | parser_exercises

# Другий ланцюг 

# Схема для плану
schemas_plan = [
    ResponseSchema(name='training_plan', description='детальний тренувальний план на тиждень')
]

parser_plan = StructuredOutputParser.from_response_schemas(schemas_plan)
instructions_plan = parser_plan.get_format_instructions()

prompt_plan = PromptTemplate.from_template(
    """
    Ти — фітнес-асистент. Побудуй тижневий план тренувань на основі:
    - Списку вправ
    - Рівня підготовки користувача
    - Кількості часу на тиждень

    Вправи: {exercises}
    Рівень підготовки: {level}
    Час на тиждень: {hours} годин

    Формат відповіді:
    {instructions}
    """,
    partial_variables={"instructions": instructions_plan}
)

chain_plan = prompt_plan | llm | parser_plan




# Крок 1: мета тренування
goal_input = "схуднення"
ex_result = chain_exercises.invoke({"goal": goal_input})

# Крок 2: рівень та час
user_level = "середній"
weekly_hours = 4

# Генеруємо тренувальний план
plan_result = chain_plan.invoke({
    "exercises": ex_result['exercises'],
    "level": user_level,
    "hours": weekly_hours
})

# Вивід
print("Вправи:")
print(ex_result['exercises'])

print("\nПлан тренувань:")
print(plan_result['training_plan'])
