from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_community.document_loaders import GithubFileLoader
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv
import os

load_dotenv()

# --- Configuración de la base de datos ---
CHROMA_PERSIST_DIR = "chroma_db"
CHROMA_COLLECTION_NAME = "portfolio_lg"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Asegúrate de que el token de acceso se cargue correctamente
ACCESS_TOKEN = os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN")

# Carga todos los archivos usando la solución anterior
loader = GithubFileLoader(
    repo="Yoseph10/AgentAI_JobSearch",
    branch="main",
    access_token=ACCESS_TOKEN,
    github_api_url="https://api.github.com",
    file_filter=lambda file_path: True,
)

documents = loader.load()

# --- Proceso para combinar los documentos ---
# Creamos una cadena vacía para guardar todo el contenido
full_content = ""

# Iteramos sobre la lista de documentos y concatenamos el contenido
for doc in documents:
    full_content += doc.page_content + "\n\n---\n\n" # Separador para distinguir entre archivos

# Opcional: puedes crear un nuevo objeto Document con todo el contenido
single_document = Document(
    page_content=full_content,
    metadata={"source": "Yoseph10/AgentAI_JobSearch", "repo_content": "true"}
)


def add_single_document_to_chroma(document: Document):
    """
    Fragmenta un solo documento y lo añade a la base de datos Chroma existente.
    """
    try:
        # 1. Inicializar el fragmentador de texto
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)

        # 2. Fragmentar el documento
        chunks = splitter.split_documents([document])
        print(f"Se generaron {len(chunks)} fragmentos del documento.")

        # 3. Inicializar el modelo de embeddings
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

        # 4. Conectar a la base de datos Chroma existente
        vectordb = Chroma(
            persist_directory=CHROMA_PERSIST_DIR,
            embedding_function=embeddings,
            collection_name=CHROMA_COLLECTION_NAME
        )

        # 5. Añadir los nuevos fragmentos a la colección existente
        vectordb.add_documents(chunks)
        print(f"Éxito: Se añadieron {len(chunks)} fragmentos a la base de datos Chroma.")

    except Exception as e:
        print(f"Error al añadir el documento a Chroma: {e}")

# Ejecutar la función con el documento que deseas añadir
if __name__ == "__main__":

    #Añadir el documento único a la base de datos Chroma
    #add_single_document_to_chroma(single_document)


    #Añado el código para cargar un PDF y añadirlo a Chroma
    pdf_path = "./data_sources/Portafolio Extendido.pdf"

    # 1. Cargar el PDF
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # 2. Dividir el contenido en fragmentos (chunks)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)

    # 3. Crear los embeddings y cargar los chunks en Chroma
    # Asegúrate de que la variable de entorno OPENAI_API_KEY esté configurada
    embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")

    # Si es la primera vez, se creará la base de datos en el directorio especificado
    vector_store = Chroma.from_documents(
        documents=chunks,
        persist_directory=CHROMA_PERSIST_DIR,
        embedding=embeddings_model,
        collection_name=CHROMA_COLLECTION_NAME
    )
    # Opcional: Para verificar que todo se cargó correctamente
    print(f"Se han cargado {len(chunks)} fragmentos en la base de datos Chroma.")
