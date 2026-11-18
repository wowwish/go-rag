package rag

import (
	"context"
	"fmt"
	"rag-course/llm"
	"rag-course/vector"
)

const defaultTopK = 5 // number of hits chosen to retrieve from the vector store

type Options struct {
	TopK     int
	Rewriter *Rewriter
}

type Retriever struct {
	embedder llm.Embedder
	store    vector.Store
	rewriter *Rewriter
	topK     int
}

func New(embedder llm.Embedder, store vector.Store, opts Options) *Retriever {
	topK := opts.TopK
	if topK <= 0 {
		topK = defaultTopK
	}
	return &Retriever{
		embedder: embedder,
		store:    store,
		rewriter: opts.Rewriter,
		topK:     topK,
	}
}

// Function to generate top hits for a given
func (r *Retriever) Retrieve(ctx context.Context, history []llm.Message) (string, error) {
	// build the query string
	query := r.buildQuery(ctx, history)
	if query == "" {
		return "", nil
	}

	// get the embedding vector of the query
	vecs, err := r.embedder.Embed(ctx, []string{query})
	if err != nil {
		return "", fmt.Errorf("ember query: %w", err)
	}
	if len(vecs) == 0 {
		return "", nil
	}

	// get the topK hits that match query embedding vector
	hits, err := r.store.Query(ctx, vecs[0], r.topK)
	if err != nil {
		return "", fmt.Errorf("vector query: %w", err)
	}
	if len(hits) == 0 {
		return "", nil
	}

	// return the formatted content from the topK hits as a single string
	return formatContext(hits), nil
}

// Function to Compact history of messages into a single string if a rewriter is used
func (r *Retriever) buildQuery(ctx context.Context, history []llm.Message) string {
	if r.rewriter != nil {
		if q, err := r.rewriter.Rewrite(ctx, history); err == nil && q != "" {
			return q
		}
	}
	return lastUserMessage(history)
}

func lastUserMessage(history []llm.Message) string {
	for i := len(history) - 1; i >= 0; i-- {
		if history[i].Role == "user" {
			return history[i].Content
		}
	}
	return ""
}
