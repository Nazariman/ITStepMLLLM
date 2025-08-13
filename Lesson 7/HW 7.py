import os
import re
import json
import hashlib
from pathlib import Path
from typing import List, Dict

import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


# ===================== КОНФІГ (можна змінити у сайдбарі) =====================
DEFAULT_PERSIST_DIR = "chroma_db"
DEFAULT_COLLECTION = "lesson_rag_docs"
DEFAULT_IDS_JSON = "data/lesson_rag/ids.json"
DEFAULT_EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# ===================== УТИЛІТИ =====================
def get_vectorstore(persist_dir: str, collection_name: str, model_name: str):
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    vs = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_dir,
    )
    return vs

def split_into_blocks(text: str) -> List[str]:
    """Розбиває текст на блоки за двома+ порожніми рядками."""
    parts = re.split(r"(?:\r?\n)\s*(?:\r?\n)\s*(?:\r?\n)+", text)
    return [p.strip() for p in parts if p.strip()]

def title_of_block(block: str) -> str:
    """Повертає перший непорожній рядок як назву блоку."""
    for line in block.splitlines():
        t = line.strip()
        if t:
            return t[:180]
    return "Untitled"

def stable_id_from_text(text: str) -> str:
    """Детерміністичний ID: MD5 від вмісту блоку."""
    return hashlib.md5(text.encode("utf-8")).hexdigest()

def append_ids_json(ids_path: Path, items: List[Dict]):
    payload = {"items": []}
    if ids_path.exists():
        try:
            payload = json.loads(ids_path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict) or "items" not in payload or not isinstance(payload["items"], list):
                payload = {"items": []}
        except Exception:
            payload = {"items": []}
    payload["items"].extend(items)
    ids_path.parent.mkdir(parents=True, exist_ok=True)
    ids_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

def add_blocks_to_db(vs: Chroma, blocks: List[str], filename_for_meta: str, ids_json_path: Path):
    texts, metadatas, ids = [], [], []
    for b in blocks:
        bid = stable_id_from_text(b)
        metadatas.append({"file": filename_for_meta, "block_title": title_of_block(b)})
        texts.append(b)
        ids.append(bid)

    # upsert (Chroma сам оновить/додасть за id)
    vs.add_texts(texts=texts, metadatas=metadatas, ids=ids)
    vs.persist()

    # у JSON пишемо лише нові ідентифікатори з метаданими
    items = [{"id": i, "file": m["file"], "block_title": m["block_title"]} for i, m in zip(ids, metadatas)]
    append_ids_json(Path(ids_json_path), items)

    return ids, metadatas


# ===================== UI =====================
st.set_page_config(page_title="Vector DB Admin", page_icon="🧩", layout="wide")
st.title("🧩 Vector DB Admin (Chroma)")

# Сайдбар — налаштування
with st.sidebar:
    st.header("Налаштування БД")
    persist_dir = st.text_input("Chroma persist_directory", DEFAULT_PERSIST_DIR)
    collection_name = st.text_input("Collection name", DEFAULT_COLLECTION)
    ids_json = st.text_input("Шлях до ids.json", DEFAULT_IDS_JSON)
    emb_model = st.text_input("Модель ембедінгів", DEFAULT_EMB_MODEL)
    st.caption("Переконайтеся, що цей набір налаштувань відповідає тому, що ви використовували на попередньому занятті.")

vs = get_vectorstore(persist_dir, collection_name, emb_model)

tab_get, tab_add, tab_search = st.tabs(["📄 Отримати документ", "➕ Додати документ", "🔎 Пошук (перевірка)"])

# -------------------- Отримати документ --------------------
with tab_get:
    st.subheader("Отримати документ за ID")
    get_id = st.text_input("Введіть ID документа", "")
    if st.button("Завантажити документ"):
        if not get_id:
            st.warning("Введіть ID.")
        else:
            try:
                # У Chroma немає прямого get_by_id у LangChain-обгортці — але можна через similarity на сам ID (як костиль),
                # або через внутрішній клієнт. Найпростіше: пошук exact id через raw API:
                collection = vs._collection  # low-level chromadb.Collection
                res = collection.get(ids=[get_id])
                ids_ret = res.get("ids", [])
                if ids_ret:
                    meta = res.get("metadatas", [{}])[0] or {}
                    cont = res.get("documents", [""])[0]
                    st.success("Документ знайдено.")
                    st.write(f"**ID:** `{get_id}`")
                    st.write(f"**Файл:** {meta.get('file', '—')}")
                    st.write(f"**Назва блоку:** {meta.get('block_title', '—')}")
                    st.text_area("Вміст", cont, height=300)
                else:
                    st.error("Документ з таким ID не знайдено.")
            except Exception as e:
                st.error(f"Помилка: {e}")

# -------------------- Додати документ --------------------
with tab_add:
    st.subheader("Додати новий документ")
    st.markdown("**Варіант 1:** завантажити TXT‑файл")
    up = st.file_uploader("Оберіть TXT", type=["txt"])
    st.markdown("**Варіант 2:** вставити текст вручну")
    manual_text = st.text_area("Текст (за бажанням)", height=200, placeholder="Вставте сюди текст, якщо не завантажуєте файл…")
    filename_for_meta = st.text_input("Назва файлу для метаданих", value="manual_input.txt")

    if st.button("Додати до БД"):
        try:
            raw_text = ""
            used_filename = filename_for_meta

            if up is not None:
                raw_text = up.read().decode("utf-8", errors="ignore")
                used_filename = up.name
            elif manual_text.strip():
                raw_text = manual_text
            else:
                st.warning("Додайте файл або текст.")
                st.stop()

            blocks = split_into_blocks(raw_text) if ("\n\n\n" in raw_text or "\r\n\r\n\r\n" in raw_text) else [raw_text.strip()]
            if not blocks:
                st.warning("Порожній вміст.")
                st.stop()

            ids, metas = add_blocks_to_db(vs, blocks, used_filename, ids_json)
            st.success(f"Додано {len(ids)} блок(и/ів).")
            with st.expander("Показати додані ID"):
                for i, m in zip(ids, metas):
                    st.write(f"- `{i}` — **{m.get('block_title','Untitled')}** (файл: {m.get('file','—')})")

        except Exception as e:
            st.error(f"Помилка: {e}")

# -------------------- Пошук (перевірка) --------------------
with tab_search:
    st.subheader("Семантичний пошук (перевірка)")
    q = st.text_input("Пошуковий запит", value="What actions are prohibited by Google Terms of Service?")
    k = st.slider("Скільки результатів показати", 1, 10, 5)
    if st.button("Шукати"):
        try:
            docs = vs.similarity_search(q, k=k)
            if not docs:
                st.info("Нічого не знайдено.")
            else:
                for d in docs:
                    meta = d.metadata or {}
                    st.markdown(f"**{meta.get('block_title','Untitled')}**  \n*Файл:* {meta.get('file','—')}")
                    st.code(d.page_content[:1200])
        except Exception as e:
            st.error(f"Помилка: {e}")
