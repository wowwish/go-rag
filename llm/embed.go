package llm

import (
	"context"
	"fmt"

	"github.com/openai/openai-go/v3"
)

// An interface for the Embedder which will convert query and text chunks into vectors
type Embedder interface {
	Embed(ctx context.Context, texts []string) ([][]float32, error) // Returns a slice of slices (slice of vectors)
}

func (c *Client) Embed(ctx context.Context, texts []string) ([][]float32, error) {
	if len(texts) == 0 {
		return nil, nil
	}

	// Batching a slice of texts for embedding. The OpenAI SDK will not match order of individual elements in the batch.
	// Hence, we need to use some idx to match embedding output to text input. Also, the OpenAI SDK returns embeddings in
	// float64 type which we quantize to float32 type in our application to reduce disk usage in our vector store.
	resp, err := c.sdk.Embeddings.New(ctx, openai.EmbeddingNewParams{
		Model: c.cfg.Model,
		Input: openai.EmbeddingNewParamsInputUnion{OfArrayOfStrings: texts},
	})
	if err != nil {
		return nil, err
	}

	// Sanity check of embedded response
	if len(resp.Data) != len(texts) {
		return nil, fmt.Errorf("embeddings: expected %d vectors; got %d", len(texts), len(resp.Data))
	}

	vecs := make([][]float32, len(texts))
	for _, d := range resp.Data {
		idx := int(d.Index) // Index to match output to input
		if idx < 0 || idx >= len(vecs) {
			return nil, fmt.Errorf("embeddings: index %d out of range", idx)
		}
		vec := make([]float32, len(d.Embedding)) // An individual vector embedding for a single text piece
		for i, f := range d.Embedding {
			vec[i] = float32(f)
		}
		vecs[idx] = vec // Populate our embedding vectors map using idx as the key to help with ordering the batch
	}

	return vecs, nil
}
