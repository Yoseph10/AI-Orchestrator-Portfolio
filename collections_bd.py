import os
import chromadb
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

# --- Configuración de la base de datos ---
CHROMA_PERSIST_DIR = "chroma_db"

def list_chroma_collections():
    """
    Lista todas las colecciones dentro del directorio de persistencia de Chroma.
    """
    try:
        # Crea un cliente de Chroma en lugar de una instancia de LangChain Chroma.
        # Esto te da acceso directo a las funcionalidades de la base de datos.
        client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)

        # Usa el método list_collections del cliente para ver los nombres.
        collections = client.list_collections()

        if not collections:
            print("No se encontraron colecciones en el directorio:", CHROMA_PERSIST_DIR)
            return

        print("Colecciones encontradas en la base de datos Chroma:")
        for collection in collections:
            # Los objetos de la colección tienen una propiedad 'name'
            print(f"- {collection.name}")

    except Exception as e:
        print(f"Error al listar las colecciones: {e}")

if __name__ == "__main__":
    list_chroma_collections()
