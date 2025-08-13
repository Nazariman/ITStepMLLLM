import os
import json
import uuid
import re
from pathlib import Path
from typing import List, Tuple, Dict

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# ====== КОНФІГ ======
TXT_PATH = Path("data/lesson_rag/huge_file.txt")
IDS_JSON = Path("data/lesson_rag/ids.json")

# Каталог існуючої Chroma-бази (НЕ видаляємо його; Chroma сам допише нові записи)
PERSIST_DIR = Path("chroma_db")           # змініть, якщо у вас інший шлях
COLLECTION_NAME = "lesson_rag_docs"       # змініть на свою колекцію, якщо треба

# Яка модель для ембедінгів (зручна, легка, без ключів)
EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# ====== УТИЛІТИ ======
def read_blocks(txt_path: Path) -> List[str]:
    """
    Читає файл та розбиває на блоки.
    Між блоками — ДВА порожніх рядки: розділювач ~ '\n\n\n' (запасом через regex).
    Повертає список рядків-блоків (без зайвих пробілів).
    """
    text = txt_path.read_text(encoding="utf-8", errors="ignore")
    # сплітимо за двома й більше порожніми рядками (переноси можуть різнитись)
    parts = re.split(r"(?:\r?\n)\s*(?:\r?\n)\s*(?:\r?\n)+", text)
    blocks = [p.strip() for p in parts if p.strip()]
    return blocks


def title_of_block(block: str) -> str:
    """
    Повертає перший непорожній рядок як назву блоку.
    """
    for line in block.splitlines():
        t = line.strip()
        if t:
            return t[:180]  # обріжемо дуже довгі заголовки
    return "Untitled"


def build_docs(blocks: List[str], file_name: str) -> Tuple[List[str], List[Dict], List[str]]:
    """
    З блоків формує:
    - texts: вміст сторінок
    - metadatas: метадані {file, block_title}
    - ids: uuid4 для кожного блоку
    """
    texts, metas, ids = [], [], []
    for b in blocks:
        t = title_of_block(b)
        doc_id = str(uuid.uuid4())
        texts.append(b)
        metas.append({"file": file_name, "block_title": t})
        ids.append(doc_id)
    return texts, metas, ids


def append_ids_json(ids: List[str], metas: List[Dict], json_path: Path):
    """
    Додає нові ID у JSON. Формат:
    {
      "items": [
        {"id": "...", "file": "...", "block_title": "..."},
        ...
      ]
    }
    """
    payload = {"items": []}
    if json_path.exists():
        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict) or "items" not in payload or not isinstance(payload["items"], list):
                payload = {"items": []}
        except Exception:
            payload = {"items": []}

    for i, m in zip(ids, metas):
        payload["items"].append({"id": i, "file": m.get("file"), "block_title": m.get("block_title")})

    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def get_vectorstore(persist_dir: Path, collection: str):
    """
    Повертає існуючу/створює Chroma‑колекцію з EMB_MODEL.
    """
    embeddings = HuggingFaceEmbeddings(model_name=EMB_MODEL)
    vs = Chroma(
        collection_name=collection,
        embedding_function=embeddings,
        persist_directory=str(persist_dir),
    )
    return vs


def quick_verify(vs: Chroma, query: str = "What are key restrictions in Google Terms of Service?"):
    """
    Друк топ‑3 результатів для ручної перевірки.
    """
    print("\n[Verify] Top-3 matches for query:\n", query)
    docs = vs.similarity_search(query, k=3)
    for i, d in enumerate(docs, 1):
        meta = d.metadata or {}
        print(f"\n{i}. {meta.get('block_title', 'Untitled')}")
        print(f"   File: {meta.get('file')}")
        snippet = (d.page_content[:300] + "…") if len(d.page_content) > 300 else d.page_content
        print("   Snippet:", snippet.replace("\n", " ")[:300])


def main():
    if not TXT_PATH.exists():
        raise FileNotFoundError(f"Не знайдено файл: {TXT_PATH}")

    print(f"Читаю файл: {TXT_PATH}")
    blocks = read_blocks(TXT_PATH)
    print(f"Блоків знайдено: {len(blocks)}")

    texts, metas, ids = build_docs(blocks, file_name=TXT_PATH.name)
    print(f"Готую до індексації: {len(texts)} документів")

    vs = get_vectorstore(PERSIST_DIR, COLLECTION_NAME)

    # upsert у Chroma: додаємо нові документи + метадані + id
    vs.add_texts(texts=texts, metadatas=metas, ids=ids)
    vs.persist()
    print(f"✅ Додано до колекції '{COLLECTION_NAME}'. Папка БД: {PERSIST_DIR}")

    # оновлюємо ids.json
    append_ids_json(ids, metas, IDS_JSON)
    print(f"✅ Оновлено JSON з ID: {IDS_JSON}")

    # швидка перевірка семплом запиту
    quick_verify(vs, query="What actions are prohibited by Google Terms of Service?")


if __name__ == "__main__":
    main()
