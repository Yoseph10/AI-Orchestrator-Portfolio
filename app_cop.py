# APP USING ELASTICSEARCH AND LLAMAINDEX
from fastapi import FastAPI
import os
import re
from typing import TypedDict, Annotated, Literal, Dict

from dotenv import load_dotenv
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from llama_index.vector_stores.elasticsearch import ElasticsearchStore as le
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.vector_stores import (
    MetadataFilter,
    MetadataFilters,
    FilterOperator,
)

import smtplib
from email.message import EmailMessage
import traceback

from psycopg_pool import AsyncConnectionPool
from psycopg.rows import dict_row
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

# --- 1. Configuración Inicial ---
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CV_PATH = "data_sources/cv.pdf"
SENDER_EMAIL = os.getenv("SENDER_EMAIL")
SENDER_PASSWORD = os.getenv("SENDER_PASSWORD")
CALENDLY_URL = os.getenv("CALENDLY_URL")
# Elastic credentials
es_password = os.getenv("ES_PASSWORD")
es_url = os.getenv("ES_URL")
es_user = os.getenv("ES_USER")

# OPTIMIZATION: Define global variables for expensive objects
# These will be initialized once at startup.
embed_model: OpenAIEmbedding | None = None
vector_store: le | None = None
vector_index: VectorStoreIndex | None = None
llm_classifier: ChatOpenAI | None = None
llms_with_tools: Dict[str, ChatOpenAI] = {}
graph = None
memory: AsyncPostgresSaver | None = None

# --- 2. Herramientas del Agente (Tools) ---
@tool
def search_elastic(query: str, k: int = 4, filters: dict = None) -> str:
    """
    Busca información relevante en el CV y otros documentos del portafolio,
    opcionalmente filtrando por metadatos como el 'file_name'.
    Úsalo para responder preguntas sobre experiencia, proyectos, habilidades, y formación.
    """
    print(f"[Search Elastic] Iniciando búsqueda con query: '{query}', k: {k}, filters: {filters}")

    # OPTIMIZATION: Use the pre-initialized vector_index
    if not vector_index:
        return "Error: El índice de búsqueda no está inicializado. Por favor, revisa la configuración del servidor."

    llama_filters = None
    if filters:
        filter_list = []
        for key, value in filters.items():
            if isinstance(value, dict) and "eq" in value:
                value = value["eq"]
            print(f"Applying filter - Key: {key}, Value: {value}")
            filter_list.append(MetadataFilter(key=key, value=value, operator=FilterOperator.EQ))

        if filter_list:
            llama_filters = MetadataFilters(filters=filter_list)

    retriever = vector_index.as_retriever(filters=llama_filters, similarity_top_k=k)

    try:
        results = retriever.retrieve(query)
    except Exception as e:
        print("[ERROR] Ocurrió un error en la recuperación:")
        traceback.print_exc()
        return f"Error al realizar la búsqueda: {e}"

    output = "\n\n".join([
        f"[{i+1}] {node.node.text.strip()}\nFuente: {node.node.metadata.get('source', 'N/A')}"
        for i, node in enumerate(results)
    ])

    return output if output else "No se encontraron resultados relevantes."

# --- [Las herramientas send_cv y schedule_call permanecen sin cambios] ---
@tool
def send_cv(email_to: str) -> str:
    """
    Envía una copia del CV a la dirección de correo electrónico proporcionada.
    Solo se debe usar cuando un reclutador pide explícitamente el CV.
    """
    if not os.path.exists(CV_PATH):
        return f"Error: El archivo del CV no se encuentra en '{CV_PATH}'."

    if not email_to:
        return "Error: No se proporcionó una dirección de correo electrónico."

    try:
        msg = EmailMessage()
        msg["Subject"] = "CV de tu candidato | Yoseph Ayala"
        msg["From"] = SENDER_EMAIL
        msg["To"] = email_to
        msg.set_content("Adjunto mi Curriculum Vitae para tu revisión. ¡Gracias!")

        with open(CV_PATH, "rb") as attachment:
            file_data = attachment.read()
            file_name = os.path.basename(CV_PATH)

        msg.add_attachment(file_data, maintype="application", subtype="pdf", filename=file_name)

        with smtplib.SMTP("smtp.gmail.com", 587) as smtp:
            smtp.starttls()
            smtp.login(SENDER_EMAIL, SENDER_PASSWORD)
            smtp.send_message(msg)

        return f"¡Éxito! El CV ha sido enviado a {email_to}."
    except smtplib.SMTPAuthenticationError:
        return "Error de autenticación. Por favor, revisa tu correo y contraseña de aplicación en el archivo .env."
    except Exception as e:
        return f"Error al enviar el correo: {e}"

@tool
def schedule_call(nombre: str, email: str, asunto: str) -> str:
    """
    Proporciona un enlace a Calendly para que un usuario pueda agendar una llamada.
    Solo se debe usar cuando un usuario solicita explícitamente agendar una reunion.
    """
    if not CALENDLY_URL:
        return "Error: La URL de Calendly no está configurada. Por favor, contacta al desarrollador."

    message = (
        f"Claro, {nombre}. Para agendar una reunión, por favor usa mi enlace de Calendly. "
        "Ahí podrás ver mis horarios disponibles y elegir el que mejor te convenga. "
        f"El enlace es: {CALENDLY_URL}"
    )
    return message
# --- [Fin de las herramientas sin cambios] ---


# --- Definición de conjuntos de herramientas por rol ---
tools_reclutador = [search_elastic, send_cv]
tools_cliente = [search_elastic, schedule_call]
tools_alumno = [search_elastic]
tools_otro = [search_elastic, schedule_call]

TOOLS_BY_ROLE = {
    "reclutador": tools_reclutador,
    "cliente": tools_cliente,
    "alumno": tools_alumno,
    "otro": tools_otro,
    "pregunta": tools_otro,
}

# --- Descripciones amigables de las herramientas ---
TOOL_DESCRIPTIONS = {
    "reclutador": "Como reclutador, puedes preguntarme sobre la **experiencia, habilidades y proyectos** del candidato. También puedo **enviarte su CV** por correo si lo necesitas.",
    "cliente": "Como cliente, puedo responder a tus preguntas, ayudarte a entender cómo puedo solucionar tus problemas y **agendar una llamada** si estás listo para discutir un proyecto.",
    "alumno": "Como estudiante, puedes preguntarme sobre **cualquier tema en el CV** relacionado con cursos, proyectos o formación para inspirarte.",
    "otro": "Te doy la bienvenida. Puedes preguntarme sobre la **experiencia profesional, habilidades o proyectos** del portafolio. También puedo **agendar una llamada** para ti.",
}

# --- 3. Estado del Grafo (Graph State) ---
class GraphState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    user_type: Literal["reclutador", "alumno", "cliente", "pregunta", "otro"]
    last_user_type: str

# --- 4. Nodos del Grafo (Graph Nodes) ---
def classifier_node(state: GraphState):
    conversation_history = state["messages"]
    formatted_history = "\n".join([f"{msg.type}: {msg.content}" for msg in conversation_history])

    # OPTIMIZATION: Use the pre-initialized classifier LLM
    if not llm_classifier:
        raise ValueError("El LLM clasificador no está inicializado.")

    prompt = f"""
    Eres un clasificador de intención. Analiza la CONVERSACIÓN COMPLETA para determinar la intención del ÚLTIMO mensaje del usuario.
    Responde SOLO con una de estas categorías: 'reclutador', 'alumno', 'cliente', 'pregunta', 'otro'.
    Si el último mensaje es una pregunta general o no define un rol claro, usa 'pregunta'.

    CONVERSACIÓN:
    {formatted_history}

    Basado en la conversación, ¿cuál es la categoría del último mensaje 'human'?
    Categoría:
    """

    classification = llm_classifier.invoke(prompt).content.strip().lower()
    cleaned_classification = re.sub(r'[^a-z]', '', classification)
    if cleaned_classification not in TOOLS_BY_ROLE.keys():
        cleaned_classification = "otro"

    print(f"[Classifier] Intención detectada: {cleaned_classification}")
    return {"user_type": cleaned_classification}

def agent_node(state: GraphState):
    user_type = state.get("user_type", "otro")
    last_user_type = state.get("last_user_type", None)

    initial_message = ""
    if user_type != last_user_type:
        initial_message = f"¡Perfecto! {TOOL_DESCRIPTIONS.get(user_type, '')}\n"

    filter_logic = ""
    # ... [La lógica de los prompts no cambia] ...
    if user_type == "reclutador":
        filter_logic = (
             "Para responder preguntas sobre la experiencia laboral y el curriculum del candidato, utiliza la herramienta `search_elastic`"
             ".Asegúrate de incluir el filtro `filters={'file_name': 'cv.pdf'}` en la llamada a la herramienta para buscar solo en el CV."
        )
    elif user_type in ["cliente", "alumno"]:
        filter_logic = (
            "Para responder preguntas sobre proyectos o formación académica, usa la herramienta `search_elastic`. "
            "Siempre incluye el filtro `filters={'file_name': 'Portafolio Extendido.pdf'}` en la llamada a la herramienta para obtener resultados detallados."
        )
    else:
        filter_logic = (
            "Para este tipo de usuario, utiliza la herramienta `search_elastic` sin ningún filtro para buscar en todos los documentos disponibles."
        )

    system_prompts = {
        "reclutador": f"""Eres un asistente profesional que representa al dueño de este portafolio. Es decir, eres Yoseph Ayala el consultor en IA y Data Science.
            Tu objetivo es destacar su experiencia y habilidades para conseguir un empleo. {filter_logic}.
            Solo usa la herramienta de enviar CV (`send_cv`) si el usuario lo pide explícitamente.
            Si no cuentas con  la información, responde directamente que no cuentas con la información. Sé amable y profesional.""",
        "cliente": f"""Eres un asistente comercial y consultor que actua como si fuese Yoseph Ayala. Tu objetivo es entender las necesidades del cliente,
            explicar cómo los servicios o productos pueden ayudarle, y facilitar el contacto o agendar una reunión. Sé proactivo y servicial.
            {filter_logic} para vender los servicios del portafolio. Utiliza la herramienta de agendar llamada (`schedule_call`) cuando el usuario lo solicite.
            Si no cuentas con  la información, responde directamente que no cuentas con la información. Sé amable y profesional.""",
        "alumno": f"""Eres Yoseph Ayala, un profesor. Tu objetivo es resolver dudas sobre cursos o formación que ofreces. {filter_logic}.
            Si no cuentas con  la información, responde directamente que no cuentas con la información. Sé amable y profesiona""",
        "otro": f"""Eres un asistente conversacional general que actua como Yoseph Ayala. Responde de forma amable y profesional. {filter_logic}.
            Si la pregunta no es sobre el portafolio, responde de forma conversacional.
            La herramienta de agendar llamada (`schedule_call`) solo debe usarse si el usuario lo pide.
            Si no cuentas con  la información, responde directamente que no cuentas con la información. Sé amable y profesional.""",
        "pregunta": f"""Eres un asistente conversacional que actua como Yoseph Ayala. Tu objetivo es responder la pregunta del usuario. {filter_logic}.
            Si no cuentas con  la información, responde directamente que no cuentas con la información. Sé amable y profesional.""",
    }
    # ... [Fin de la lógica de prompts] ...

    prompt_content = system_prompts.get(user_type, system_prompts["otro"])
    system_message = SystemMessage(content=prompt_content)

    # OPTIMIZATION: Use the pre-configured LLM with tools for the current user type
    llm_with_tools = llms_with_tools.get(user_type)
    if not llm_with_tools:
        raise ValueError(f"El LLM para el rol '{user_type}' no está inicializado.")

    messages_for_llm = [system_message] + state["messages"]
    response = llm_with_tools.invoke(messages_for_llm)

    final_messages = []
    if initial_message:
        final_messages.append(AIMessage(content=initial_message))
    final_messages.append(response)

    return {"messages": final_messages, "last_user_type": user_type}

# --- [clarification_node, router, y should_continue permanecen sin cambios] ---
def clarification_node(state: GraphState):
    return {"messages": [AIMessage(content="¡Hola! Para poder ayudarte mejor, ¿me podrías decir si eres un reclutador, un posible cliente, un alumno o tienes otra consulta?")], "last_user_type": "pregunta"}

def router(state: GraphState) -> Literal["agent", "clarification"]:
    user_type = state["user_type"]
    last_user_type = state.get("last_user_type")

    if last_user_type and user_type == "pregunta" and last_user_type != "pregunta":
        state["user_type"] = last_user_type
        print(f"[Router] Manteniendo rol '{last_user_type}'.")
        return "agent"

    if user_type == "pregunta" and not last_user_type:
        print("[Router] No se ha definido un rol. Yendo a aclaración.")
        return "clarification"

    print(f"[Router] Clasificación exitosa: '{user_type}'. Yendo a agente.")
    return "agent"

def should_continue(state: GraphState) -> Literal["tools", "__end__"]:
    last_message = state["messages"][-1]
    return "tools" if last_message.tool_calls else "__end__"
# --- [Fin de las secciones sin cambios] ---


# --- 6. Construcción del Grafo ---
all_tools = tools_reclutador + tools_cliente + tools_alumno + tools_otro
tool_node = ToolNode(all_tools)

builder = StateGraph(GraphState)
builder.add_node("classifier", classifier_node)
builder.add_node("agent", agent_node)
builder.add_node("tools", tool_node)
builder.add_node("clarification", clarification_node)
builder.set_entry_point("classifier")

builder.add_conditional_edges(
    "classifier",
    router,
    {"agent": "agent", "clarification": "clarification"}
)
builder.add_conditional_edges("agent", should_continue)
builder.add_edge("tools", "agent")
builder.add_edge("clarification", END)

app = FastAPI()

# --- 7. Compilación y Configuración de la Memoria ---
@app.on_event("startup")
async def startup_event():
    global memory, graph, embed_model, vector_store, vector_index, llm_classifier, llms_with_tools

    print("Iniciando servidor y configurando recursos...")

    # --- 1. Inicializar conexión a Elasticsearch y LlamaIndex ---
    print("Inicializando modelo de embeddings...")
    embed_model = OpenAIEmbedding(model="text-embedding-3-small")

    print("Conectando a Elasticsearch...")
    vector_store = le(
        index_name="ai_portafolio",
        es_url=es_url,
        es_user=es_user,
        es_password=es_password
    )

    print("Cargando VectorStoreIndex...")
    vector_index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        embed_model=embed_model
    )
    print("¡Índice de LlamaIndex listo!")

    # --- 2. Inicializar modelos de lenguaje (LLMs) ---
    print("Inicializando LLMs...")
    llm_classifier = ChatOpenAI(model="gpt-4o", temperature=0.0)

    # Pre-compilar un LLM con herramientas para cada rol
    for role, tools in TOOLS_BY_ROLE.items():
        llm = ChatOpenAI(model="gpt-4o", temperature=0.2)
        llms_with_tools[role] = llm.bind_tools(tools)
    print("¡LLMs con herramientas listos!")

    # --- 3. Configurar memoria y compilar el grafo ---
    print("Configurando memoria persistente (Postgres)...")
    pool = AsyncConnectionPool(
        conninfo=f"postgres://{os.getenv('PSQL_USERNAME')}:{os.getenv('PSQL_PASSWORD')}"
        f"@{os.getenv('PSQL_HOST')}:{os.getenv('PSQL_PORT')}/{os.getenv('PSQL_DATABASE')}"
        f"?sslmode={os.getenv('PSQL_SSLMODE')}",
        max_size=20
    )
    conn = await pool.getconn()
    memory = AsyncPostgresSaver(conn=conn)
    # await memory.setup() # Descomentar solo para la configuración inicial de la tabla

    print("Compilando el grafo...")
    graph = builder.compile(checkpointer=memory)
    print("¡Aplicación lista para recibir peticiones!")

@app.on_event("shutdown")
async def shutdown_event():
    # En una aplicación real, aquí podrías cerrar conexiones de la pool si es necesario.
    print("Apagando servidor... listo.")

@app.post("/chat")
async def chat_endpoint(user_input: str):
    if graph is None:
        return {"error": "El grafo no está inicializado. El servidor podría estar arrancando."}

    config = {"configurable": {"thread_id": "3"}} # Usa un ID único para cada usuario
    all_messages = []

    async for paso in graph.astream(
        {"messages": [HumanMessage(content=user_input)]},
        config,
        stream_mode="values"
    ):
        all_messages.extend(paso["messages"])

    final_assistant_message = all_messages[-1].content if all_messages else "No se generó respuesta."
    return {"response": final_assistant_message}
