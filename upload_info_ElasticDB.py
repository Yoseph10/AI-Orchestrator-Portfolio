#UPLOAD DATA TO ELASTICSEARCH DB USING LLAMA INDEX

from llama_index.core.extractors import TitleExtractor
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.elasticsearch import ElasticsearchStore as le
from llama_index.core import StorageContext
from llama_index.core import VectorStoreIndex
from dotenv import load_dotenv
import os

load_dotenv()

# --- Configuración de la base de datos ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
es_elast = os.getenv("ES_PASSWORD")
es_url = os.getenv("ES_URL")

# Paso 1: Cargar documentos locales
documents = SimpleDirectoryReader("./data_sources2").load_data()

# Paso 2: Definir pipeline de ingesta

embedding_model = OpenAIEmbedding(model="text-embedding-3-small")

pipeline = IngestionPipeline(
    transformations=[
        SentenceSplitter(chunk_size=1500, chunk_overlap=50),
        TitleExtractor(),
        embedding_model,
    ]
)

# Paso 3: Ejecutar pipeline para obtener nodos embebidos
nodes = pipeline.run(documents = documents)

# Paso 4: Almacenar en ElasticSearch

vector_store = le(
    es_url=es_url,
    es_user="elastic",
    es_password=es_elast,
    index_name="ai_portafolio",
)

storage_context = StorageContext.from_defaults(vector_store=vector_store)

# Guardamos usando VectorStoreIndex
index = VectorStoreIndex(nodes, storage_context=storage_context)
