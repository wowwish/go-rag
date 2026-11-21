package web

import (
	"context"
	"embed"
	"encoding/json"
	"errors"
	"fmt"
	"html/template"
	"io"
	"io/fs"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"rag-course/ingest"
	"rag-course/llm"
	"rag-course/rag"
	"rag-course/vector"
	"strings"
	"time"

	"github.com/go-chi/chi/middleware"
	"github.com/go-chi/chi/v5"
)

// Use the GO-embed directive to add static files and binaries into this GO application at compile time

//go:embed templates/*.gohtml
var templatesFS embed.FS

const maxUploadBytes = 10 << 20 // Bit manipulation on decimal value 2 to get decimal value of 2^20.

type uploadResponse struct {
	Source string `json:"source"`
	Bytes  int    `json:"bytes"`
	Chunks int    `json:"chunks"`
}

type Options struct {
	Addr             string
	SystemPromptFile string
	Title            string
	Store            vector.Store
	ProcessDir       string
	ImagesDir        string
}

type Server struct {
	client     *llm.Client
	embedder   *llm.Client
	retriever  *rag.Retriever
	store      vector.Store
	processDir string
	imagesDir  string
	tpl        *template.Template
	system     string // System Prompt
	title      string
}

// Function to create a new web server interface.
func New(client, embedder *llm.Client, retriever *rag.Retriever, opts Options) (*Server, error) {
	// Parse the template for the web server landing page
	tpl, err := template.ParseFS(templatesFS, "templates/*.gohtml")
	if err != nil {
		return nil, fmt.Errorf("parse templates: %w", err)
	}

	title := opts.Title
	if title == "" {
		title = "RAG Chat"
	}

	// Create a new web server instance
	return &Server{
		client:     client,
		embedder:   embedder,
		store:      opts.Store,
		retriever:  retriever,
		processDir: opts.ProcessDir,
		imagesDir:  opts.ImagesDir,
		tpl:        tpl,
		system:     readSystemPrompt(opts.SystemPromptFile),
		title:      title,
	}, nil
}

// Handler method for API routes of the Web server
func (s *Server) Routes() http.Handler {
	r := chi.NewRouter() // Create a new route handler to be used by the server
	r.Use(middleware.Recoverer)
	r.Get("/chat", s.handleChatPage)               // add route "/chat" to the server and the corresponding route handler function
	r.Post("/api/chat/stream", s.handleChatStream) // add route to get back response to user query with history from the LLM model
	// following the Server-Sent-Events (SSE) protocol to stream the tokens as chunks over the long-lived HTTP web server connection.
	// The server responds with Content-Type: text/event-stream over standard HTTP. Messages are UTF-8 text blocks separated by two
	// newlines ("\n\n"). Common fields include data, event, id, and retry. Browsers use the native JavaScript EventSource
	// interface to connect and listen for incoming messages. If the connection drops, EventSource automatically reconnects and
	// can pass the last received event id. However, EventSource only supports GET requests. Since we use POST requests in our case,
	// we implement a custome version of the Protocol using the fetch() API in Javascript.
	r.Post("/api/upload", s.handleUpload) // add route to handle post requests fired by detected file uploads in the Web UI
	if s.imagesDir != "" {
		r.Post("/api/upload/image", s.handleUploadImage) // add route to handle image upload post requests
		fs := http.FileServer(http.Dir(s.imagesDir))
	}
	return r
}

// Type declaration to hold chat messages as a JSON serializable instance
type chatRequest struct {
	Messages []llm.Message `json:"messages"`
}

// Method to handle file upload logic
func (s *Server) handleUpload(w http.ResponseWriter, r *http.Request) {
	// Without a configured vector store, there is nowhere to process and store text chunks from the uploaded file for grounding
	// the LLM model.
	if s.store == nil {
		http.Error(w, "ingest is not configured (no vector store)", http.StatusServiceUnavailable)
		return // Just exit without returning anything
	}

	// Good Practice to read only limited content from the request Body, as there is potential to bring down the server
	// by uploading massive amounts of data into such routes that expect small json for a single incoming request.
	r.Body = http.MaxBytesReader(w, r.Body, maxUploadBytes)

	if err := r.ParseMultipartForm(maxUploadBytes); err != nil {
		http.Error(w, "upload too large or malformed: "+err.Error(), http.StatusBadRequest)
		return // just exit without returning anything
	}

	file, header, err := r.FormFile("file")
	if err != nil {
		http.Error(w, "missing 'file' field: "+err.Error(), http.StatusBadRequest)
		return // just exit without returning anything
	}
	defer file.Close() // Handle resource release after use

	name := filepath.Base(header.Filename) // Get the file name
	if !ingest.IsSupported(name) {
		http.Error(w, "unsupported format", http.StatusUnsupportedMediaType)
		return // just exit without returning anything
	}

	// Read the file content
	content, err := io.ReadAll(file)
	if err != nil {
		http.Error(w, "read upload: "+err.Error(), http.StatusBadRequest)
		return // just exit without returning anything
	}

	// Generate text chunks from the file content
	chunks, err := ingest.ProcessContent(r.Context(), name, content, ingest.Options{}, s.embedder, s.store)
	if err != nil {
		log.Printf("[web] upload ingest failed for %q: %v", name, err)
		http.Error(w, "ingest failed: "+err.Error(), http.StatusInternalServerError)
		return // just exit without returning anything
	}

	// Move processed file to processed directory to prevent repeated processing of same file
	if s.processDir != "" {
		dest := filepath.Join(s.processDir, name)
		if err := os.Mkdir(s.processDir, 0o755); err != nil {
			log.Printf("[web] mkdir %s: %v", s.processDir, err)
		} else if err := os.WriteFile(dest, content, 0o644); err != nil {
			log.Printf("[web] archived %s: %v", dest, err)
		}
	}

	// Return response to UI client
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(uploadResponse{
		Source: name,
		Bytes:  len(content),
		Chunks: chunks,
	})
}

// Method to handle the route logic for streaming LLM model response as SSE.
func (s *Server) handleChatStream(w http.ResponseWriter, r *http.Request) {
	flusher, ok := w.(http.Flusher) // explicitly cast the response writer to a http flusher instance.
	// By default, Go's http.ResponseWriter buffers outgoing data (typically up to 4KB) and only transmits
	// it to the client once the buffer fills up or the HTTP handler returns. http.Flusher overrides this
	// behavior, and it accumulates data until the data is manually dumped/pushed using the .Flush() call from the instance.
	if !ok {
		http.Error(w, "streaming unsupported", http.StatusInternalServerError)
		return // Just exit after printing the error without returning anything
	}

	var req chatRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "invalid json:"+err.Error(), http.StatusBadRequest)
		return // Just exit after printing the error without returning anything
	}

	if len(req.Messages) == 0 {
		http.Error(w, "messages must not be empty", http.StatusBadRequest)
		return // Just exit after printing the error without returning anything
	}

	// Get the most recent message in the history. If it is not a message from user (latest User's query), throw error
	if last := req.Messages[len(req.Messages)-1]; last.Role != "user" {
		http.Error(w, "last message must be from user", http.StatusBadRequest)
		return // Just exit after printing the error without returning anything
	}

	history := req.Messages // Set the hostory
	if s.system != "" {
		// Append system prompt to the top of the history (as first message in history)
		history = append([]llm.Message{{Role: "system", Content: s.system}}, history...)
	}

	turn := history // Create a cloned copy of history to preserve the history before retrieved
	// relevant information to the query from the vector store is added.
	if s.retriever != nil {
		ctxText, err := s.retriever.Retrieve(r.Context(), history) // Retrieve relevant information from vector store for
		// the latest user query and then add it to the history, before sending it as context to the LLM model.
		if err != nil {
			log.Printf("[web] retrieval error: %v", err)
		} else {
			turn = withInlineContext(history, ctxText)
		}
	}

	w.Header().Set("Content-Type", "text/event-stream") // Required Header for Server Side Events Protocol
	w.Header().Set("Cache-Control", "no-cache")         // Ensure no Caching of streamed data as required by SSE Protocol Standard
	w.Header().Set("Connection", "keep-alive")          // Use a single HTTP connection, avoid repetitive TCP three-way handshakes.
	// This is required for sending data packets smoothly as SSE.
	w.Header().Set("X-Accel-Buffering", "no") // Disable Buffering of data chunks in Reverse Proxies like Nginx, so that
	// data chunks are sent immediately as they arrive.
	flusher.Flush() // Manually send the data chunks as they arrive. When the user submits the web server form, the query is
	// added to history and streamed via SSE to the server route that handles the submission.

	// Anonymous function to help with data chunking logic for UI to split apart and utilize
	send := func(event, data string) {
		if event != "" {
			fmt.Fprintf(w, "event: %s\n", event)
		}
		fmt.Fprintf(w, "data: %s\n\n", data)
		flusher.Flush()
	}

	_, err := s.client.ChatStream(r.Context(), turn, func(delta string) {
		enc, _ := json.Marshal(delta)
		send("delta", string(enc))
	})
	if err != nil {
		enc, _ := json.Marshal(err.Error())
		send("error", string(enc))
		return
	}
	send("done", `""`)
}

// Function to combine user query and its relevant embedded context information retrieved from the vector store,
// as the last message into history.
func withInlineContext(history []llm.Message, ctxText string) []llm.Message {
	if len(history) == 0 || ctxText == "" {
		return history // just return original history when no relevant context has been retrieved from vector store to add
	}
	last := history[len(history)-1]
	if last.Role != "user" {
		return history // return original history if the latest message in history is not from the user
	}
	out := make([]llm.Message, len(history))
	// create a clone copy of history, add the user's query along with the retrieved context for the query from the vector
	// store as the last message into the clone of history and return it
	copy(out, history)
	out[len(out)-1] = llm.Message{
		Role:    "user",
		Content: ctxText + "\n\n--- Question ---\n\n" + last.Content,
	}
	return out
}

// Function to handle requests routed to "/chat" path of the server
func (s *Server) handleChatPage(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "text/html; charset=utf8")
	if err := s.tpl.ExecuteTemplate(w, "chat.gohtml", map[string]any{
		"Title": s.title,
		// "CaptionEnabled": s.client.HasVision(),
	}); err != nil {
		log.Printf("[web] template error: %v", err)
	}
}

// Function to run the Server instance
func (s *Server) Run(ctx context.Context, addr string) error {
	srv := &http.Server{
		Addr:              addr,
		Handler:           s.Routes(), // This is the server level handler, hence we provide the high level route handler function
		ReadHeaderTimeout: 5 * time.Second,
	}

	// Use a channel to monitor for server errors during its Runtime
	errCh := make(chan error, 1)
	// Run the server instance in the background as a go routine
	go func() {
		err := srv.ListenAndServe() // Blocking
		if !errors.Is(err, http.ErrServerClosed) {
			errCh <- err // If the error is not server shutdown or closing related, add it to the error surveillance channel
		}
		errCh <- nil // Send nil for error if server Run was successful and if the server closed by normal shutdown command.
	}()

	// Use a select statement to watch for Server instance termination due to normal shutdown command or due to an error
	// and handle the cases appropriately (Graceful Shutdown)
	select {
	case <-ctx.Done():
		shutDownCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()                // handle context resource cleanup when the server Run is shutdown without errors.
		_ = srv.Shutdown(shutDownCtx) // Force normal shutdown of server when the runtime context of the server given to this
		// function as an argument is providing a Done signal.
		return nil
	// Handle any runtime error encountered by the server (other than forced shutdown or close) by returning the error
	case err := <-errCh:
		return err
	}
}

// Function to parse the system prompts file
func readSystemPrompt(path string) string {
	if path == "" {
		return ""
	}

	data, err := os.ReadFile(path)
	if errors.Is(err, fs.ErrNotExist) {
		return ""
	}
	return strings.TrimSpace(string(data))
}
