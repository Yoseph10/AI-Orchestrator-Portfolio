import os
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

# --- 1. Configuración Inicial ---
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CHROMA_PERSIST_DIR = "chroma_db"
CHROMA_COLLECTION_NAME = "portfolio_lg"

# --- 2. Cargar y Ver el Contenido de los Documentos ---
def view_chroma_documents_with_content():
    """
    Carga la colección de Chroma y muestra el contenido y metadatos de los documentos.
    """
    if not os.path.exists(CHROMA_PERSIST_DIR):
        print(f"Error: El directorio de persistencia '{CHROMA_PERSIST_DIR}' no existe.")
        return

    try:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        # Cargar la base de datos Chroma existente
        vectordb = Chroma(
            persist_directory=CHROMA_PERSIST_DIR,
            embedding_function=embeddings,
            collection_name=CHROMA_COLLECTION_NAME
        )

        if vectordb._collection.count() == 0:
            print("La colección de Chroma está vacía. No hay documentos para mostrar.")
            return

        print(f"Colección '{CHROMA_COLLECTION_NAME}' encontrada con {vectordb._collection.count()} documentos.")
        print("-" * 50)

        # Usar vectordb.get() para obtener todos los documentos, metadatos e IDs
        # Incluimos "documents" para obtener el contenido de cada fragmento
        results = vectordb._collection.get(include=["documents", "metadatas"])

        documents = results.get("documents", [])
        metadatas = results.get("metadatas", [])
        ids = results.get("ids", [])

        print("Contenido y metadatos de los documentos:")
        for i in range(len(ids)):
            print(f"--- Documento {i+1} (ID: {ids[i]}) ---")
            print("Metadatos:", metadatas[i])
            print("Contenido:")
            print(documents[i])
            print("\n")

    except Exception as e:
        print(f"Ocurrió un error: {e}")

if __name__ == "__main__":
    view_chroma_documents_with_content()
