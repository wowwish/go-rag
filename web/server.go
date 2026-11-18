package web

import (
	"context"
	"embed"
	"errors"
	"fmt"
	"html/template"
	"io/fs"
	"log"
	"net/http"
	"os"
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

// Handle API routes for the Web server
func (s *Server) Routes() http.Handler {
	r := chi.NewRouter() // Create a new route handler to be used by the server
	r.Use(middleware.Recoverer)
	r.Get("/chat", s.handleChatPage) // add route "/chat" to the server and the corresponding route handler function
	return r
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
