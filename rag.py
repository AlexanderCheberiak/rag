import os
from pathlib import Path

from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


INPUT_DATA_DIR = "markdown_docs"
CHROMA_DB_DIR = "chroma_db"
MODEL_NAME = "qwen2.5:7b"        
EMBED_MODEL = "bge-m3"                 

def build_vector_database():
    if not os.path.exists(INPUT_DATA_DIR):
        os.makedirs(INPUT_DATA_DIR, exist_ok=True)
        print(f"Папку '{INPUT_DATA_DIR}' не знайдено. Створено порожню.")
        return None

    print(f"Зчитування Markdown-файлів з папки '{INPUT_DATA_DIR}'...")
    
    loader = DirectoryLoader(
        INPUT_DATA_DIR, 
        glob="**/*.md", 
        loader_cls=TextLoader, 
        loader_kwargs={'encoding': 'utf-8', 'autodetect_encoding': True}
    )
    documents = loader.load()
    
    if not documents:
        print("Папка з документами порожня")
        return None

    print(f"Оброблено документів: {len(documents)}. Розбиття на чанки...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300)
    chunks = text_splitter.split_documents(documents)

    print(f"Створення векторних ембеддінгів...")
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    
    print("Запис чанків у локальну базу ChromaDB...")
    vector_db = Chroma.from_documents(chunks, embeddings, persist_directory=CHROMA_DB_DIR)
    return vector_db

def main():
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    
    if os.path.exists(CHROMA_DB_DIR) and os.listdir(CHROMA_DB_DIR):
        print("Знайдено готову векторну базу. Завантаження...")
        vector_db = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embeddings)
    else:
        print("Векторна база не знайдена. Ініціалізація нової бази...")
        vector_db = build_vector_database()
        if not vector_db:
            return

    retriever = vector_db.as_retriever(search_kwargs={"k": 6})

    prompt = ChatPromptTemplate.from_template("""Ти — досвідчений та вимогливий викладач університету. 
Твоє завдання — допомагати студентам засвоювати матеріал, спираючись ВИКЛЮЧНО на надані фрагменти тексту з підручників (Контекст).

ТВОЇ СУВОРІ ПРАВИЛА:
1. НІКОЛИ не давай прямих відповідей, готових розв'язків чи написаного коду.
2. Використовуй наданий контекст, щоб сформувати підказку або навідне запитання, яке змусить студента подумати.
3. Якщо відповіді немає в Контексті, прямо скажи: "У нашому підручнику про це не йдеться. Спробуй перефразувати запитання".
4. Спілкуйся виключно українською мовою.

Контекст з підручника:
{context}

Запитання студента:
{question}

Твоя відповідь-підказка:"""
)

    llm = ChatOllama(model=MODEL_NAME, temperature=0.3)

    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    print("\n" + "="*50)
    print("Задайте своє питання або введіть 'exit' для виходу:")
    print("="*50 + "\n")

    while True:
        user_input = input("Студент: ")
        if user_input.lower() in ['exit', 'quit', 'вихід']:
            print("Сесію завершено.")
            break
        if not user_input.strip():
            continue
            
        print("ШІ-Асистент: ", end="", flush=True)
        for chunk in rag_chain.stream(user_input):
            print(chunk, end="", flush=True)
        print("\n" + "-"*50)

if __name__ == "__main__":
    main()