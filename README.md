# Installation and Usage

## Setup Local LLM Model

Install Ollama from https://ollama.com/download/linux
```
curl -fsSL https://ollama.com/install.sh | sh
```

Then, pull the models required for the RAG application (If using local models with the App)
```
ollama pull gemma3
ollama pull nomic-embed-text
ollama pull mistral-small3.1
```

Alternatively, you can setup a Cloud API Key from Ollama as the env var **OPENAI_API_KEY** in the `.env` file.


## Setup Vector Store

To set start the Postgres vector DB for Putting in reference documents, use:
```
docker compose up -d
docker exec -it rag-course-postgres psql -U rag
# Copy SQL commands from create-table.sql
exit
```
To bring down the DB, use:
```
docker compose down -v
```

## Start the App

Then, start the RAG app (GO automatically searches for main.go in the directory path specified)
```
cd go-rag
go run ./cmd/rag
```

Once the App is running, you can configure an ingestion directory into which you can drop text and markdown files which will be automatically ingested and embedded into the vector store for use. Then, you can place the files in this directory for grounding the LLM model with relevant information.

## Web Server

The application also spawns an interactive Web App that can be accessed using the browser. Go to the link provided by the application startup in a browser to use the Web App.

The Web App allows you to:
* Upload text files with contextual information that can be addded to the RAG knowledgebase
* Upload image files