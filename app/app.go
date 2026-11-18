package app

import (
	"context"
	"log"
	"os"
	"rag-course/chat"
	"rag-course/config"
	"rag-course/ingest"
	"rag-course/llm"
	"rag-course/rag"
	"rag-course/vector"
	"rag-course/vector/pgvector"
	"rag-course/web"
	"sync"
)

func Run(parent context.Context, cfg config.Config) error {
	logger := log.New(os.Stderr, "[rag] ", log.LstdFlags)

	// Create a cancellable context from a parent context
	ctx, cancel := context.WithCancel(parent)
	defer cancel() // Release the created context when this function ends

	// Create a new LLM Client
	client := llm.New(cfg)

	// Create a new Embedder
	embedder := llm.NewEmbedder(cfg)

	// Open the vector store
	store, err := openStore(ctx, cfg)
	if err != nil {
		logger.Printf("vector store disabled: %v", err)
	}

	var wg sync.WaitGroup // used to make this Run function call wait until some other Goroutine call completes.
	if store != nil {
		// call a function as a GO-Routine and add it to the wait group
		// This anonymous function will keep watching an ingest directory for new files or file changes
		// until an error is encountered, or the context used by it is cancelled.
		wg.Go(func() {
			opts := ingest.Options{
				SourceDir:    cfg.IngestDir,
				ProcessedDir: cfg.ProcessedDir,
			}

			if err := ingest.Watch(ctx, opts, embedder, store, logger); err != nil && ctx.Err() == nil {
				logger.Printf("watcher stopped: %v", err)
			}
		})
		logger.Printf("watching %s for new documents", cfg.IngestDir)
	}

	if store != nil {
		defer store.Close()
		logger.Printf("vector store ready")
	}

	// Get a Retriever type instance, which means we also need a Rewriter instance
	var retriever *rag.Retriever
	// If we donot have a vector store, we cannot get a retriever instance, so we need to check for the presence of a store
	if store != nil {
		retriever = rag.New(embedder, store, rag.Options{
			TopK:     5, // Use hardcoded number as topK
			Rewriter: rag.NewRewriter(client),
		})
	}

	// Setup a new Web Server instance when the HTTP address is provided to run it in the config of the app
	if cfg.HttpAddr != "" {
		srv, err := web.New(client, embedder, retriever, web.Options{
			Addr:             cfg.HttpAddr,
			SystemPromptFile: cfg.SystemPromptFile,
			Store:            store,
			ProcessDir:       cfg.ProcessedDir,
			ImagesDir:        cfg.ImageDir,
		})
		if err != nil {
			logger.Printf("web server disabled: %v", err)
		} else {
			wg.Go(func() {
				if err := srv.Run(ctx, cfg.HttpAddr); err != nil && ctx.Err() == nil {
					logger.Printf("web server stopped: %v", err)
				}
			})
			logger.Printf("web chat at http://localhost%s/chat", cfg.HttpAddr)
		}

	}

	replErr := chat.RunREPL(ctx, client, retriever, chat.Options{
		SystemPromptFile: cfg.SystemPromptFile,
	})

	cancel()  // Cancel the context used by the watcher to stop the watcher
	wg.Wait() // Wait for the watcher to end
	return replErr
}

func openStore(ctx context.Context, cfg config.Config) (vector.Store, error) {
	if cfg.DatabaseURL == "" {
		return nil, nil
	}
	s, err := pgvector.New(ctx, pgvector.Options{
		DSN:          cfg.DatabaseURL,
		EmbeddingDim: cfg.EmbeddingDim,
	})
	if err != nil {
		return nil, err
	}

	return s, nil
}
