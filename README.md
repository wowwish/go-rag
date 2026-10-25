# Installation and Usage

## Setup Local LLM Model

Install Ollama from https://ollama.com/download/linux
```
curl -fsSL https://ollama.com/install.sh | sh
```

Then, pull the latest gemma3 model (If using local model with the App)
```
ollama pull gemma3
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