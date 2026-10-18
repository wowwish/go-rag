package config

import (
	"os"

	"github.com/joho/godotenv" // required for loading environment variables from a .env file
)

type Config struct {
	BaseURL          string // any openai compatible server uri
	APIKey           string // the API key for authenticating with the OpenAI compatible server, or a dummy value for local LLM
	Model            string // the model to be used by the OpenAI compatible server or local LLM
	SystemPromptFile string // the path to the system prompt file used by the REPL
}

func Load() Config {
	_ = godotenv.Load()

	cfg := Config{
		BaseURL:          os.Getenv("OPENAI_BASE_URL"),
		APIKey:           os.Getenv("OPENAI_API_KEY"),
		Model:            os.Getenv("OPENAI_MODEL"),
		SystemPromptFile: os.Getenv("SYSTEM_PROMPT_FILE"),
	}

	if cfg.BaseURL == "" {
		cfg.BaseURL = "https://api.openai.com/v1"
	}

	if cfg.Model == "" {
		cfg.Model = "gpt-4o-mini"
	}

	return cfg
}
