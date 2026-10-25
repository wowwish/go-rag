// Package ingest takes in documents from a source directory and puts them
// in the vector store back end.
// Five Steps:
// 1. READ
// 2. CHUNK
// 3. EMBED
// 4. DELETE
// 5. UPSERT

package ingest

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"rag-course/llm"
	"rag-course/vector"
	"strconv"
	"strings"
	"time"
)

const (
	defaultChunkSize    = 1000
	defaultChunkOverlap = 100
)

type Options struct {
	SourceDir    string
	ProcessedDir string
	ChunkSize    int
	ChunkOverlap int
}

// Per-file pipeline
func processOne(ctx context.Context, path string, opts Options, embedder llm.Embedder, store vector.Store) error {
	if !supportedFormat(path) {
		return fmt.Errorf("unsupported format: %s", filepath.Ext(path))
	}

	raw, err := os.ReadFile(path)
	if err != nil {
		return fmt.Errorf("read: %w", err)
	}

	_, err = processContent(ctx, filepath.Base(path), raw, opts, embedder, store)
	return err
}

func processContent(ctx context.Context, source string, content []byte, opts Options, embedder llm.Embedder, store vector.Store) (int, error) {
	if embedder == nil {
		return 0, errors.New("embedder is required")
	}

	if store == nil {
		return 0, errors.New("vector store is required")
	}

	// Get the filename from the input file path
	base := filepath.Base(source)
	if !supportedFormat(base) {
		return 0, fmt.Errorf("unsupported format: %s", filepath.Ext(base))
	}

	size := opts.ChunkSize
	if size <= 0 {
		size = defaultChunkSize
	}

	overlap := opts.ChunkOverlap
	if overlap <= 0 {
		overlap = defaultChunkOverlap
	}

	text := strings.TrimSpace(string(content))
	if text == "" {
		return 0, errors.New("file is empty")
	}

	chunks := chunk(text, size, overlap)
	if len(chunks) == 0 {
		return 0, errors.New("no chunks produced")
	}

	vectors, err := embedder.Embed(ctx, chunks)
	if err != nil {
		return 0, fmt.Errorf("embed: %w", err)
	}

	if len(vectors) != len(chunks) {
		return 0, fmt.Errorf("embed: got %d vectors for %d chunks", len(vectors), len(chunks))
	}

	// We want to delete pre-existing chunks for this source so a re-ingestion starts from a clean slate
	// This helps in situations where data with changes is dropped into the source directory as it will
	// automatically be picked up and updated in our vector store.
	if err := store.DeleteBySource(ctx, base); err != nil {
		return 0, fmt.Errorf("clear previous chunks: %w", err)
	}

	ingestedAt := time.Now().UTC().Format(time.RFC3339) // Get current timestamp in a specific format
	docs := make([]vector.Document, len(chunks))
	for i, c := range chunks {
		docs[i] = vector.Document{
			ID:      fmt.Sprintf("%s%d", base, i), // We just use the filename and chunk index as the unique ID for a
			Content: c,
			Metadata: map[string]string{
				"source":      base,
				"chunk_index": strconv.Itoa(i),
				"chunks":      strconv.Itoa(len(chunks)),
				"ingested_at": ingestedAt,
			},
			Embedding: vectors[i], // embedded vector at current index i
		}
	}

	if err := store.Upsert(ctx, docs); err != nil {
		return 0, err
	}

	return len(chunks), nil
}

// Function to check if a file is of the supported format for ingestion
func supportedFormat(path string) bool {
	switch strings.ToLower(filepath.Ext(path)) {
	case ".txt", ".md", ".markdown":
		return true
	}
	return false
}
