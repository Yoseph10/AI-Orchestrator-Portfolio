import os
import json
import time
import uuid
import re
from typing import TypedDict, Annotated, Literal, List, Any, Dict

from dotenv import load_dotenv
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_core.tools import tool
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver


import sys
import locale

import smtplib
from email.message import EmailMessage



# Force Python to use UTF-8 for I/O streams
if sys.stdout.encoding != 'utf-8':
    sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)
if sys.stderr.encoding != 'utf-8':
    sys.stderr = open(sys.stderr.fileno(), mode='w', encoding='utf-8', buffering=1)
if sys.stdin.encoding != 'utf-8':
    sys.stdin = open(sys.stdin.fileno(), mode='r', encoding='utf-8', buffering=1)

# Ensure the system's default locale is also set to UTF-8
try:
    locale.setlocale(locale.LC_ALL, 'en_US.utf8')
except locale.Error:
    try:
        locale.setlocale(locale.LC_ALL, 'C.UTF-8')
    except locale.Error:
        pass


# --- 1. Configuración Inicial ---
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CHROMA_PERSIST_DIR = "chroma_db"
CHROMA_COLLECTION_NAME = "portfolio_lg"
CV_PATH = "data_sources/cv.pdf"
SENDER_EMAIL = os.getenv("SENDER_EMAIL")
SENDER_PASSWORD = os.getenv("SENDER_PASSWORD")
CALENDLY_URL = os.getenv("CALENDLY_URL")

# --- 2. Herramientas del Agente (Tools) ---
@tool
def search_chroma(query: str, k: int = 4, filters: dict = None) -> List[Dict[str, Any]]:
    """
    Busca información relevante en el CV y otros documentos del portafolio,
    opcionalmente filtrando por metadatos como el 'source'.
    Úsalo para responder preguntas sobre experiencia, proyectos, habilidades, y formación.
    """
    try:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        vectordb = Chroma(persist_directory=CHROMA_PERSIST_DIR, embedding_function=embeddings, collection_name=CHROMA_COLLECTION_NAME)
        if vectordb._collection.count() == 0:
            return [{"error": "La base de datos Chroma está vacía."}]

        # Realiza la búsqueda, aplicando el filtro si está presente
        if filters:
            docs = vectordb.similarity_search(query, k=k, filter=filters)
        else:
            docs = vectordb.similarity_search(query, k=k)

        return [{"source": d.metadata.get("source", "N/A"), "text": d.page_content} for d in docs]
    except Exception as e:
        return [{"error": f"Error buscando en Chroma: {str(e)}"}]

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

    # Nota: También podrías pasar el nombre, email y asunto como parámetros URL para que Calendly los precargue.
    # Por ejemplo: f"{CALENDLY_URL}?name={nombre}&email={email}&a1={asunto}"
    # Sin embargo, el LLM debería poder manejar esta lógica por sí mismo en la mayoría de los casos.

    return message

# --- Definición de conjuntos de herramientas por rol ---
tools_reclutador = [search_chroma, send_cv]
tools_cliente = [search_chroma, schedule_call]
tools_alumno = [search_chroma]
tools_otro = [search_chroma, schedule_call]

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
            "Utiliza la herramienta de búsqueda (`search_chroma`)"
            "con los filtros `{'source': 'data_sources/cv.pdf'}` o `{'source': 'Yoseph10/AgentAI_JobSearch'}` "
            "para buscar información relevante sobre mi CV o proyectos. Si la información es sobre el proyecto del portafolio `AgentAI_JobSearch`, usa el segundo filtro. Si es sobre mi experiencia laboral o habilidades, usa el primer filtro."
        )
    elif user_type in ["cliente", "alumno"]:
        filter_logic = (
            "Utiliza la herramienta de búsqueda (`search_chroma`)"
            "con el filtro `{'source': './data_sources/Portafolio Extendido.pdf'}`"
        )
    else:
        # No se aplica ningún filtro para 'otro' o 'pregunta'
        filter_logic = (
            "Para este tipo de usuario, utiliza la herramienta `search_chroma` sin ningún filtro para buscar en todos los documentos disponibles."
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
    user_type = state["user_type"]
    return "clarification" if user_type == "pregunta" else "agent"


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


# --- 7. Compilación y Configuración de la Memoria ---
memory = MemorySaver()
graph = builder.compile(checkpointer=memory)


# --- 8. Funciones Auxiliares para el CLI ---
def ingest_pdf_to_chroma(pdf_path: str):
    if not os.path.exists(pdf_path):
        return f"[Error] El archivo no existe: {pdf_path}"
    try:
        print(f"[Ingest] Cargando {pdf_path}...")
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        chunks = splitter.split_documents(docs)
        print(f"[Ingest] {len(chunks)} fragmentos generados. Creando embeddings...")
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        vectordb = Chroma.from_documents(chunks, embeddings, persist_directory=CHROMA_PERSIST_DIR, collection_name=CHROMA_COLLECTION_NAME)
        return f"[Ingest] Éxito: {len(chunks)} fragmentos indexados en la colección '{CHROMA_COLLECTION_NAME}'."
    except Exception as e:
        return f"[Ingest] Error: {e}"

def ensure_chroma_db_is_ready():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    try:
        vectordb = Chroma(persist_directory=CHROMA_PERSIST_DIR, embedding_function=embeddings, collection_name=CHROMA_COLLECTION_NAME)
        if vectordb._collection.count() > 0:
            print(f"[RAG] Base de datos Chroma existente con {vectordb._collection.count()} documentos. No es necesaria la indexación.")
            return True
        else:
            print("[RAG] Base de datos Chroma vacía o no existe. Iniciando indexación del CV.")
            result = ingest_pdf_to_chroma(CV_PATH)
            print(result)
            return "Éxito" in result
    except Exception as e:
        print(f"[RAG] Error al verificar la base de datos: {e}")
        return False


# --- 9. Interfaz de Línea de Comandos (CLI) ---
def main():
    print("Portafolio Multiagente con LangGraph (CLI)")
    print("Comandos: /exit")

    if not ensure_chroma_db_is_ready():
        print("[Error fatal] No se pudo inicializar la base de datos de conocimientos. El agente no podrá responder preguntas sobre el CV.")
        return

    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    while True:
        try:
            user_input = input('> ')
        except (EOFError, KeyboardInterrupt):
            print("\n[Agente]: Adiós.")
            break

        if not user_input:
            continue
        if user_input.lower() in ["/exit", "/quit"]:
            print("[Agente]: ¡Hasta luego!")
            break

        for step in graph.stream({"messages": [HumanMessage(content=user_input)]}, config, stream_mode="values"):
            msg = step["messages"][-1]
            print(msg.content, end='', flush=True)
        print()

if __name__ == "__main__":
    main()
