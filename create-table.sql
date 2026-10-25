CREATE EXTENSION IF NOT EXISTS vector; -- enable extension vector

-- Create the vector store (Table)
CREATE TABLE documents (
    id text PRIMARY KEY,
    content text NOT NULL, -- This column is only there for educational purposes. In production, it is not required.
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb, -- We will store the compressed binary JSON form of a chunk of content in this column to save space.
    embedding vector(768) NOT NULL, -- This is the part that the LLM uses to retrieve relevant content in RAG system
    created_at timestamp with time zone NOT NULL DEFAULT now()
);

CREATE INDEX documents_embedding_idx ON documents USING hnsw (embedding vector_cosine_ops); -- create index to speed things up