# 🧠 AI Portfolio Backend — Yoseph Ayala

This repository contains the **backend API** for the **AI Intelligent Portfolio**, designed to dynamically interact with users (recruiters, clients, or students) through specialized AI agents.

The backend provides a **serverless API** deployed on **Google Cloud Run**, built with **FastAPI**, **LangChain**, and **RAG-based reasoning** for retrieving and responding with contextual portfolio information.

---

## 🚀 Overview

The system exposes an intelligent API that:

1. Classifies the user’s type (recruiter, client, or student).
2. Activates the corresponding specialized agent.
3. Retrieves relevant data from the knowledge base (RAG).
4. Generates personalized responses or actions (e.g., send résumé, summarize experience, describe projects).

The backend powers the conversational intelligence used by the **frontend portfolio interface** hosted in a separate repository.

---

## 🧩 Architecture Overview

| Component                                | Description                                                   |
| ---------------------------------------- | ------------------------------------------------------------- |
| **FastAPI**                              | REST API exposing endpoints for AI interactions.              |
| **LangChain / LangGraph**                | Manages the orchestration of multi-agent reasoning.           |
| **RAG (Retrieval-Augmented Generation)** | Retrieves precise portfolio data using a vector store.        |
| **PostgreSQL**                           | Stores conversation history and contextual data.              |
| **Docker + Artifact Registry**           | Used for image creation and storage before deployment.        |
| **Google Cloud Run**                     | Serverless platform used to deploy and serve the backend API. |

---

## 🗂️ Repository Structure

```
AI-Portfolio-Backend/
│
├── app.py                # Main FastAPI application
├── requirements.txt      # Dependencies
├── Dockerfile            # Docker configuration for containerized deployment
├── .dockerignore
├── .gitignore
└── README.md
```

---

## ⚙️ Local Development

### 1. Clone the repository

```bash
git clone https://github.com/Yoseph10/AI-Portfolio-Backend.git
cd AI-Portfolio-Backend
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up your environment variables

Create a `.env` file in the project root:

Set your env variables. For example:

```
OPENAI_API_KEY=sk-xxxx
POSTGRES_URI=postgresql+psycopg2://user:password@host:5432/dbname
```

### 5. Run the API locally

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

Then open:
👉 [http://localhost:8000/docs](http://localhost:8000/docs) to explore the interactive Swagger documentation.

---

## 🐳 Docker + Cloud Run Deployment

### 1. Build the Docker image

```bash
docker build -t ai_portfolio_backend .
```

(Optional — rebuild without cache:)

```bash
docker build --no-cache -t ai_portfolio_backend .
```

### 2. Authenticate with Google Cloud

```bash
gcloud auth login
gcloud auth configure-docker REGION-docker.pkg.dev
```

Example:

```bash
gcloud auth configure-docker us-central1-docker.pkg.dev
```

### 3. Push the image to Artifact Registry

```bash
docker tag ai_portfolio_backend us-central1-docker.pkg.dev/PROJECT_ID/REPO_NAME/ai_portfolio_backend:latest
docker push us-central1-docker.pkg.dev/PROJECT_ID/REPO_NAME/ai_portfolio_backend:latest
```

### 4. Deploy to Cloud Run

```bash
gcloud run deploy ai-portfolio-backend \
  --image us-central1-docker.pkg.dev/PROJECT_ID/REPO_NAME/ai_portfolio_backend:latest \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --env-vars-file .env.yaml
```

---

## 📡 API Endpoints

| Endpoint   | Method | Description                                              |
| ---------- | ------ | -------------------------------------------------------- |
| `/chat`    | POST   | Main endpoint to interact with the conversational agent. |

---

## 🧠 Intelligent Components

| Component                  | Function                                                               |
| -------------------------- | ---------------------------------------------------------------------- |
| **Multi-Agent Reasoning**  | Each agent specializes in one user type (recruiter, client, student).  |
| **RAG Engine**             | Retrieves portfolio-related content (projects, education, experience). |
| **LangGraph Orchestrator** | Manages agent interactions and flow control.                           |
| **PostgreSQL Memory**      | Stores conversation states for continuity.                             |
| **OpenAI API**             | Powers natural language understanding and generation.                  |

---

## 🧱 Tech Stack

* Python 3.10+
* FastAPI
* LangChain / LangGraph
* LlamaIndex
* PostgreSQL
* Docker
* Google Cloud Run
* OpenAI API

---

## 📄 License

This project is developed for **personal and academic purposes** by **Yoseph Ayala**, as a demonstration of AI system orchestration, RAG retrieval, and serverless deployment using Google Cloud.

---

**Developed with ❤️ by Yoseph Ayala**
💡 *“An API that understands who’s asking — and answers accordingly.”*
