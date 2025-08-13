import os
from typing import List, Dict
from dotenv import load_dotenv

# LangChain Serper wrapper
from langchain_community.utilities.google_serper import GoogleSerperAPIWrapper

load_dotenv()  # підхоплює SERPER_API_KEY з .env

# ---------- "Інструмент": пошук ресторанів через Serper Places ----------
def search_restaurants(query: str, k: int = 5) -> List[Dict]:
    """
    Приймає рядок запиту та повертає список словників:
    { name, website (або None), rating (або None) }
    """
    if not os.getenv("SERPER_API_KEY"):
        raise RuntimeError("Недоступний SERPER_API_KEY. Додайте його в .env або змінні середовища.")

    # type="places" вмикає режим Google Places
    serper = GoogleSerperAPIWrapper(type="places", k=k)

    # serper.results повертає JSON з ключем 'places'
    data = serper.results(query)
    places = data.get("places", []) or []

    results = []
    for p in places:
        results.append({
            "name": p.get("title"),
            "website": p.get("website"),      # може бути відсутнім
            "rating": p.get("rating"),        # може бути відсутнім
        })
    return results

# ---------- Чат-бот (консоль) ----------
def main():
    print("🧭 Рекомендатор ресторанів (Serper Places). Введіть запит або 'exit' для виходу.")
    print("Напр.: 'італійські ресторани Київ', 'best sushi near Lviv', 'seafood Odessa'")

    while True:
        user_q = input("\nВаш запит: ").strip()
        if not user_q:
            continue
        if user_q.lower() in {"exit", "quit"}:
            print("Бувай!")
            break

        try:
            items = search_restaurants(user_q, k=7)
        except Exception as e:
            print(f"Помилка: {e}")
            continue

        if not items:
            print("Нічого не знайдено. Спробуйте перефразувати або додати місто/район.")
            continue

        print("\nТоп результатів:")
        for i, it in enumerate(items, start=1):
            name = it.get("name") or "—"
            site = it.get("website") or "—"
            rating = it.get("rating")
            rating_str = f"{rating:.1f}" if isinstance(rating, (int, float)) else "—"

            print(f"{i}. {name}")
            print(f"   Сайт: {site}")
            print(f"   Рейтинг: {rating_str}")

if __name__ == "__main__":
    main()
