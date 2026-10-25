package vector

import "context"

type Document struct {
	ID        string
	Content   string            // For educational purposes - to view the actual content chunk corresponding to an embedding vector
	Metadata  map[string]string // arbitrary structured data associated with a chunk of content
	Embedding []float32
}

// Type to store hits from a RAG vector search
type Result struct {
	Document         // Embedding the above declared struct into this struct
	Score    float32 // A measure of similarity between the query vector and the stored vector of a content chunk in the vector store
	// A good rule of thumb: score > 0.8 means strongly related and score < 0.4 is probably noise
}

// An interface that every Vector Store must implement
type Store interface {
	Upsert(ctx context.Context, docs []Document) error                          // Insert new documents or replace existing documents by ID in the vector store
	Query(ctx context.Context, embedding []float32, topK int) ([]Result, error) // Returns the top K documents similar to the searched embedding vector.
	Delete(ctx context.Context, ids []string) error                             // For deleting documents from the vector store by ID
	DeleteBySource(ctx context.Context, source string) error                    // For deleting documents that match the source field of metadata with given value in the vector store
	Close() error                                                               // To free up resources and prevent resource leaks
}
