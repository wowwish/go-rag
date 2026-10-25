package config

import (
	"os"
	"strconv"

	"github.com/joho/godotenv" // required for loading environment variables from a .env file
)

type Config struct {
	BaseURL          string // any openai compatible server uri
	APIKey           string // the API key for authenticating with the OpenAI compatible server, or a dummy value for local LLM
	Model            string // the model to be used by the OpenAI compatible server or local LLM
	SystemPromptFile string // the path to the system prompt file used by the REPL
	DatabaseURL      string // DSN for connecting to Postgres or other vector DB
	EmbeddingDim     int    // Dimensionality of the embedding used for creating vectors out of document chunks
	EmbeddingBaseURL string // base URL for the embedding model (if using cloud api from Ollama)
	EmbeddingAPIKey  string // API Key for the embedding model (Note that currently, Ollama does not have a cloud API for
	// an embedding model. But, this is in place to allow for Cloud API based embedding in future.)
	EmbeddingModel string
	IngestDir      string // Directory which will be watched to load documents from
	ProcessedDir   string
}

func Load() Config {
	_ = godotenv.Load()

	cfg := Config{
		BaseURL:          os.Getenv("OPENAI_BASE_URL"),
		APIKey:           os.Getenv("OPENAI_API_KEY"),
		Model:            os.Getenv("OPENAI_MODEL"),
		SystemPromptFile: os.Getenv("SYSTEM_PROMPT_FILE"),
		DatabaseURL:      os.Getenv("DATABASE_URL"),
		EmbeddingDim:     atoiOr(os.Getenv("EMBEDDING_DIM"), 0),
		EmbeddingBaseURL: os.Getenv("EMBEDDING_BASE_URL"),
		EmbeddingAPIKey:  os.Getenv("EMBEDDING_API_KEY"),
		EmbeddingModel:   os.Getenv("EMBEDDING_MODEL"),
		IngestDir:        os.Getenv("INGEST_DIR"),
		ProcessedDir:     os.Getenv("PROCESSED_DIR"),
	}

	if cfg.BaseURL == "" {
		cfg.BaseURL = "https://api.openai.com/v1"
	}

	if cfg.Model == "" {
		cfg.Model = "gpt-4o-mini"
	}

	// Sanity check
	if cfg.EmbeddingDim == 0 {
		cfg.EmbeddingDim = 768
	}

	// If there is no seperate base URL for the embedding model, set it to the same as the inference model URL
	if cfg.EmbeddingBaseURL == "" {
		cfg.EmbeddingBaseURL = cfg.BaseURL
		if cfg.EmbeddingAPIKey == "" {
			cfg.EmbeddingAPIKey = cfg.APIKey
		}
	}

	if cfg.EmbeddingModel == "" {
		cfg.EmbeddingModel = "nomic-embed-text"
	}

	if cfg.IngestDir == "" {
		cfg.IngestDir = "./documents"
	}

	if cfg.ProcessedDir == "" {
		cfg.ProcessedDir = "./documents/processed"
	}

	return cfg
}

func atoiOr(s string, fallback int) int {
	if s == "" {
		return fallback
	}
	n, err := strconv.Atoi(s)
	if err != nil {
		return fallback
	}
	return n
}
