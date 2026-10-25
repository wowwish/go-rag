package pgvector

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"rag-course/vector"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/pgvector/pgvector-go"
	pgxvec "github.com/pgvector/pgvector-go/pgx"
)

// Configuration object for the vector store
type Options struct {
	DSN          string // connection host to postgres
	EmbeddingDim int    // Dimension for the vector produced by the embedder
}

type Store struct {
	pool *pgxpool.Pool
}

func New(ctx context.Context, opts Options) (*Store, error) {
	if opts.DSN == "" {
		return nil, errors.New("pgvector: DSN is required!")
	}
	if opts.EmbeddingDim <= 0 {
		return nil, errors.New("pgvector: EmbeddingDim must be > 0!")
	}

	// Create a config object from parsing the postgres connection host
	// from the options type object
	cfg, err := pgxpool.ParseConfig(opts.DSN)
	if err != nil {
		return nil, fmt.Errorf("parse DSN: %w", err)
	}

	if err := ensureExtension(ctx, opts.DSN); err != nil {
		return nil, fmt.Errorf("install extension: %w", err)
	}

	// To ensure that every new connection in the database pool we have created
	// has the vector OID registered, so that pgx knows how to encode and decode types
	// between the db pool and the connecting client. Without this, the pgx driver will
	// simply return raw text representation of the vector information stored in the database.
	cfg.AfterConnect = func(ctx context.Context, conn *pgx.Conn) error {
		return pgxvec.RegisterTypes(ctx, conn)
	}

	// Create a new DB connection pool with our created config object
	pool, err := pgxpool.NewWithConfig(ctx, cfg)
	if err != nil {
		return nil, fmt.Errorf("connect: %w", err)
	}

	s := &Store{pool: pool}
	// migrate() is a convenience method that checks if our required table with required schema
	// exists, and if it does'nt, it will create it for us
	if err := s.migrate(ctx, opts.EmbeddingDim); err != nil {
		pool.Close()
		return nil, fmt.Errorf("migrate: %w", err)
	}

	return s, nil
}

func ensureExtension(ctx context.Context, dsn string) error {
	conn, err := pgx.Connect(ctx, dsn)
	if err != nil {
		return err
	}
	defer conn.Close(ctx)                                            // free up resources when this function ends execution
	_, err = conn.Exec(ctx, "CREATE EXTENSION IF NOT EXISTS vector") // This forces installation of the extension. The error
	// returned will be nil if the postgres version has support for this extension. Otherwise, an actual error will be returned.
	return err
}

func (s *Store) migrate(ctx context.Context, dim int) error {
	stmts := []string{
		fmt.Sprintf(`CREATE TABLE IF NOT EXISTS documents (
			id TEXT PRIMARY KEY,
			content TEXT NOT NULL,
			metadata JSONB NOT NULL DEFAULT '{}'::JSONB,
			embedding vector(%d) NOT NULL,
			created_at TIMESTAMPTZ NOT NULL DEFAULT now()
		)`, dim),
		`CREATE INDEX IF NOT EXISTS documents_embedding_idx
		ON documents USING HNSW(embedding vector_cosine_ops)`,
	}

	for _, q := range stmts {
		if _, err := s.pool.Exec(ctx, q); err != nil {
			return fmt.Errorf("exec %q: %w", firstLine(q), err)
		}
	}

	return nil
}

func firstLine(s string) string {
	for i := 0; i < len(s); i++ {
		if s[i] == '\n' {
			return s[:i]
		}
	}
	return s
}

// Implement all the Store interface methods to satisfy it

func (s *Store) Upsert(ctx context.Context, docs []vector.Document) error {
	if len(docs) == 0 {
		return nil // return no error if there are no documents to upsert
	}
	tx, err := s.pool.Begin(ctx) // Create a transaction with the DB
	if err != nil {
		return err
	}
	defer tx.Rollback(ctx) // Rollback the transaction if something goes wrong

	const stmt = `
		INSERT INTO documents (id, content, metadata, embedding)
		VALUES ($1, $2, $3, $4)
		ON CONFLICT DO UPDATE SET
			content = EXCLUDED.content,
			metadata = EXCLUDED.metadata,
			embedding = EXCLUDED.embedding
	`

	for _, d := range docs {
		meta, err := marshalMetadata(d.Metadata) // Serialize the Metadata map
		if err != nil {
			return fmt.Errorf("metadata for %s: %w", d.ID, err)
		}
		if _, err := tx.Exec(ctx, stmt, d.ID, d.Content, meta, pgvector.NewVector(d.Embedding)); err != nil {
			return fmt.Errorf("upsert: %s: %w", d.ID, err)
		}
	}

	return tx.Commit(ctx)

}

// Function to serialize metadata into JSON format
func marshalMetadata(m map[string]string) ([]byte, error) {
	if len(m) == 0 {
		return []byte("{}"), nil
	}
	return json.Marshal(m)
}

func unmarshalMetadata(raw []byte, dst *map[string]string) error {
	if len(raw) == 0 {
		*dst = nil
		return nil
	}
	return json.Unmarshal(raw, dst)
}

func (s *Store) Query(ctx context.Context, embedding []float32, topK int) ([]vector.Result, error) {
	if topK <= 0 {
		return nil, nil
	}

	// The <=> operator in postgres is a null-safe comparison operator that returns true for A <=> B if both A and B
	// are non-null or if both A and B are null. Normal = operator will return null when either A or B is null.
	const stmt = `
		SELECT id, content, metadata, embedding <=> $1 AS distance
		FROM documents
		ORDER BY embedding <=> $1
		LIMIT $2
	`

	rows, err := s.pool.Query(ctx, stmt, pgvector.NewVector(embedding), topK)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var results []vector.Result
	for rows.Next() {
		var (
			r        vector.Result
			metaRaw  []byte
			distance float64
		)

		if err := rows.Scan(&r.ID, &r.Content, &metaRaw, &distance); err != nil {
			return nil, err
		}
		if err = unmarshalMetadata(metaRaw, &r.Metadata); err != nil {
			return nil, fmt.Errorf("metadata for %s: %w", r.ID, err)
		}
		r.Score = float32(1 - distance)
		results = append(results, r)
	}

	return results, rows.Err()
}

func (s *Store) Delete(ctx context.Context, ids []string) error {
	if len(ids) == 0 {
		return nil
	}

	_, err := s.pool.Exec(ctx, `DELETE FROM documents WHERE id = ANY($1)`, ids)
	return err
}

func (s *Store) DeleteBySource(ctx context.Context, source string) error {
	if source == "" {
		return nil
	}

	_, err := s.pool.Exec(ctx, `DELETE FROM documents WHERE metadata->>'source' = $1`, source)
	return err
}

func (s *Store) Close() error {
	s.pool.Close()
	return nil
}
