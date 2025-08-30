#APP USING ELASTICSERACH AND LLAMAINDEX
from fastapi import FastAPI
import os
import re
from typing import TypedDict, Annotated, Literal, List, Any, Dict

from dotenv import load_dotenv
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from llama_index.vector_stores.elasticsearch import ElasticsearchStore as le
from llama_index.core import VectorStoreIndex
from llama_index.core.vector_stores import (
    MetadataFilter,
    MetadataFilters,
    FilterOperator,
)
from llama_index.core import StorageContext

import smtplib
from email.message import EmailMessage



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
#Elastic credentials
es_elast = os.getenv("ES_PASSWORD")
es_url = os.getenv("ES_URL")


# --- 2. Herramientas del Agente (Tools) ---
@tool
def search_elastic(query: str, k: int = 4, filters: dict = None) -> str:
    """
    Busca información relevante en el CV y otros documentos del portafolio,
    opcionalmente filtrando por metadatos como el 'file_name'.
    Úsalo para responder preguntas sobre experiencia, proyectos, habilidades, y formación.
    """

    print(f"[Search Elastic] Iniciando búsqueda con query: '{query}', k: {k}, filters: {filters}")

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    vector_store = le(
            es_url=es_url,
            es_user="elastic",
            es_password=es_elast,
            index_name="ai_portafolio"
    )

    storage_context_read = StorageContext.from_defaults(vector_store=vector_store)

    print("[Search Elastic] Conexión a Elasticsearch establecida.")

    try:
        index_read = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            storage_context=storage_context_read, # Este es el parámetro incorrecto
            embed_model=embeddings
        )
        print("¡El índice se cargó exitosamente!")

    except TypeError as e:
        print(f"ERROR: {e}")
        # El error te dirá algo como: "TypeError: VectorStoreIndex.from_vector_store() got an unexpected keyword argument 'storage_context'"
        # Esto te indica claramente que 'storage_context' no es un parámetro válido para esta función.
    except Exception as e:
        print(f"Ocurrió un error inesperado: {e}")

    print("[Search Elastic] VectorStoreIndex cargado correctamente.")


    if filters:
        for key, value in filters.items():
            print(f"Applying filter - Key: {key}, Value: {value}")
            llama_filters = MetadataFilters(
                filters=[
                    MetadataFilter(
                        key=key, value=value, operator=FilterOperator.TEXT_MATCH
                    )
                ]
            )

        retriever = index_read.as_retriever(filters=llama_filters, similarity_top_k=k)

        results = retriever.retrieve(query)

        print(results)

    else:
        retriever = index_read.as_retriever(search_kwargs={"k": k})
        results = retriever.retrieve(query)

    output = "\n\n".join([
        f"[{i+1}] {node.node.text.strip()}\nFuente: {node.node.metadata.get('source', 'N/A')}"
        for i, node in enumerate(results)
    ])

    return output if output else "No se encontraron resultados relevantes."


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
        # 1. Crear el objeto de mensaje
        msg = EmailMessage()
        msg["Subject"] = "CV de tu candidato | Yoseph Ayala"
        msg["From"] = SENDER_EMAIL
        msg["To"] = email_to
        msg.set_content("Adjunto mi Curriculum Vitae para tu revisión. ¡Gracias!")

        # 2. Adjuntar el archivo CV
        with open(CV_PATH, "rb") as attachment:
            file_data = attachment.read()
            file_name = os.path.basename(CV_PATH)

        msg.add_attachment(file_data, maintype="application", subtype="pdf", filename=file_name)

        # 3. Conectarse y enviar el correo usando la misma lógica que funciona
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

    # Aquí puedes personalizar el mensaje para el usuario
    message = (
        f"Claro, {nombre}. Para agendar una reunión, por favor usa mi enlace de Calendly. "
        "Ahí podrás ver mis horarios disponibles y elegir el que mejor te convenga. "
        f"El enlace es: {CALENDLY_URL}"
    )

    return message

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
    last_user_type: str # Nuevo campo para rastrear el tipo de usuario anterior


# --- 4. Nodos del Grafo (Graph Nodes) ---
def classifier_node(state: GraphState):
    conversation_history = state["messages"]
    formatted_history = "\n".join([f"{msg.type}: {msg.content}" for msg in conversation_history])
    llm = ChatOpenAI(model="gpt-4o", temperature=0.0)

    prompt = f"""
    Eres un clasificador de intención. Analiza la CONVERSACIÓN COMPLETA para determinar la intención del ÚLTIMO mensaje del usuario.
    Responde SOLO con una de estas categorías: 'reclutador', 'alumno', 'cliente', 'pregunta', 'otro'.
    Si el último mensaje es una pregunta general o no define un rol claro, usa 'pregunta'.

    CONVERSACIÓN:
    {formatted_history}

    Basado en la conversación, ¿cuál es la categoría del último mensaje 'human'?
    Categoría:
    """

    classification = llm.invoke(prompt).content.strip().lower()
    cleaned_classification = re.sub(r'[^a-z]', '', classification)
    if cleaned_classification not in TOOLS_BY_ROLE.keys():
        cleaned_classification = "otro"

    print(f"[Classifier] Intención detectada: {cleaned_classification}")
    return {"user_type": cleaned_classification}


# Modificación del agent_node
def agent_node(state: GraphState):
    user_type = state.get("user_type", "otro")
    last_user_type = state.get("last_user_type", None)

    # Generamos el mensaje de bienvenida y el prompt del sistema
    initial_message = ""
    if user_type != last_user_type:
        initial_message = f"¡Perfecto! {TOOL_DESCRIPTIONS.get(user_type, '')}\n"

    # --- Se agrega la lógica de filtros aquí ---
    filter_logic = ""
    if user_type == "reclutador":
        filter_logic = (
            "Para responder preguntas sobre la experiencia laboral y el curriculum del candidato, utiliza la herramienta `search_elastic`\
            .Asegúrate de incluir el filtro `filters={'file_name': 'cv.pdf'}` en la llamada a la herramienta para buscar solo en el CV."
        )
    elif user_type in ["cliente", "alumno"]:
        filter_logic = (
            "Para responder preguntas sobre proyectos o formación académica, usa la herramienta `search_elastic`. "
            "Siempre incluye el filtro `filters={'file_name': 'Portafolio Extendido.pdf'}` en la llamada a la herramienta para obtener resultados detallados."
        )
    else:
        # No se aplica ningún filtro para 'otro' o 'pregunta'
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

    prompt_content = system_prompts.get(user_type, system_prompts["otro"])
    system_message = SystemMessage(content=prompt_content)

    tools_for_agent = TOOLS_BY_ROLE.get(user_type, tools_otro)
    llm = ChatOpenAI(model="gpt-4o", temperature=0.2)
    llm_with_tools = llm.bind_tools(tools_for_agent)

    # El LLM necesita ver el historial completo, incluyendo el mensaje de bienvenida
    # si se ha generado.
    messages_for_llm = [system_message] + state["messages"]

    response = llm_with_tools.invoke(messages_for_llm)

    # Si hay un mensaje de bienvenida, se añade antes de la respuesta del LLM
    final_messages = []
    if initial_message:
        final_messages.append(AIMessage(content=initial_message))

    final_messages.append(response)

    # Actualizamos el estado con los nuevos mensajes y el user_type
    return {"messages": final_messages, "last_user_type": user_type}

# A este código se le debe añadir también un ajuste en el `clarification_node`
def clarification_node(state: GraphState):
    # La primera vez que el usuario llega a este nodo, el last_user_type estará vacío.
    # Así, la siguiente vez que entre al agent_node, se imprimirá el mensaje de bienvenida
    return {"messages": [AIMessage(content="¡Hola! Para poder ayudarte mejor, ¿me podrías decir si eres un reclutador, un posible cliente, un alumno o tienes otra consulta?")], "last_user_type": "pregunta"}

# --- 5. Lógica de Enrutamiento (Router) ---
def router(state: GraphState) -> Literal["agent", "clarification"]:
    # Obtener la clasificación actual y la anterior (si existe)
    user_type = state["user_type"]
    last_user_type = state.get("last_user_type")

    # Si ya se ha establecido un rol y el clasificador no lo cambió a "pregunta",
    # entonces continuamos con el agente sin volver a preguntar.
    if last_user_type and user_type == "pregunta" and last_user_type != "pregunta":
        # Se asume que la nueva pregunta es parte del mismo rol, a menos que se indique lo contrario.
        # Volvemos a asignar el user_type anterior para no perder el contexto.
        state["user_type"] = last_user_type
        print(f"[Router] Manteniendo rol '{last_user_type}'.")
        return "agent"

    # Si el usuario es 'pregunta' y es la primera vez que se interactúa, se va a aclaración.
    if user_type == "pregunta" and not last_user_type:
        print("[Router] No se ha definido un rol. Yendo a aclaración.")
        return "clarification"

    # Si la clasificación es una de las opciones válidas, va al agente.
    print(f"[Router] Clasificación exitosa: '{user_type}'. Yendo a agente.")
    return "agent"


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

def should_continue(state: GraphState) -> Literal["tools", "__end__"]:
    last_message = state["messages"][-1]
    return "tools" if last_message.tool_calls else "__end__"

builder.add_conditional_edges("agent", should_continue)
builder.add_edge("tools", "agent")
builder.add_edge("clarification", END)


app = FastAPI()


# --- 7. Compilación y Configuración de la Memoria ---
#memory = MemorySaver()
#graph = builder.compile(checkpointer=memory)


memory: AsyncPostgresSaver | None = None
graph = None


@app.on_event("startup")
async def startup_event():
    global memory
    pool = AsyncConnectionPool(
        conninfo=f"postgres://{os.getenv('PSQL_USERNAME')}:{os.getenv('PSQL_PASSWORD')}"
        f"@{os.getenv('PSQL_HOST')}:{os.getenv('PSQL_PORT')}/{os.getenv('PSQL_DATABASE')}"
        f"?sslmode={os.getenv('PSQL_SSLMODE')}",
        max_size=20,
        kwargs={
            "autocommit": True,
            "prepare_threshold": 0,
            "row_factory": dict_row,
        },
    )
    conn = await pool.getconn()   # obtenemos una conexión
    memory = AsyncPostgresSaver(conn)
    # ⚠️ aquí ya puedes compilar el grafo con memoria persistente

    # IMPORTANT: You need to call .setup() the first time you're using your memory
    await memory.setup()

    global graph
    graph = builder.compile(checkpointer=memory)


@app.on_event("shutdown")
async def shutdown_event():
    print("Apagando servidor... listo.")


@app.post("/chat")
async def chat_endpoint(user_input: str):
    config = {"configurable": {"thread_id": "1"}} # Use a unique ID for each user in production
    all_messages = []
    async for paso in graph.astream(
        {"messages": [HumanMessage(content=user_input)]},
        config,
        stream_mode="values"
    ):
        all_messages.extend(paso["messages"])

    final_assistant_message = all_messages[-1].content if all_messages else "No se generó respuesta."
    return {"response": final_assistant_message}
