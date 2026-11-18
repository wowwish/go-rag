package chat

import (
	"bufio"
	"context"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"rag-course/llm"
	"rag-course/rag"
	"strings"
	"sync"
	"time"
)

// options for a single REPL or Chat session
type Options struct {
	SystemPromptFile string
}

// A means to read and write messages from console interactively.
func RunREPL(ctx context.Context, client *llm.Client, retriever *rag.Retriever, opts Options) error {
	in := bufio.NewScanner(os.Stdin)               // object to read from standard input (console)
	in.Buffer(make([]byte, 0, 64*1024), 1024*1024) // Set the initial buffer size to 64 KB and the maximum buffer size to 1 MB
	// This is how much content the scanner can read at once, allowing for large inputs without running into buffer
	// limitation.

	// history is a slice of llm.Message elements representing the conversation history.
	history, err := seedHistory(opts.SystemPromptFile)
	if err != nil {
		return err
	}

	fmt.Println("Chat session started. Type Q to quit.")
	// Infinite loop for REPL
	for {
		fmt.Print("\n> ") // Prompt for user input
		if !in.Scan() {
			// If scanning failed due to an error, return it. Otherwise, return nil to indicate end of input.
			if err := in.Err(); err != nil {
				return err
			}
			return nil
		}

		input := strings.TrimSpace(in.Text())
		if input == "" {
			continue // Skip empty input
		}

		// Check if the user wants to quit the chat session. EqualFold is used for case-insensitive comparison.
		if strings.EqualFold(input, "q") || strings.EqualFold(input, "/exit") || strings.EqualFold(input, "exit") || strings.EqualFold(input, "quit") {
			fmt.Println("Goodbye.")
			return nil // Exit the REPL loop and end the chat session
		}

		// Append the user's message to the conversation history
		history = append(history, llm.Message{
			Role:    "user",
			Content: input,
		})

		// Start the spinner to indicate that the program is processing the user's input.
		spin := startSpinner("thinking...")
		var stopOnce sync.Once // Ensures the spinner is stopped only once. sync.Once is a
		// concurrency primitive that ensures a function is only executed once, even if called from multiple goroutines.
		// Here, the ChatStream method uses the history slice to generate streaming response chunks and we can
		// use callback functions to add additional behavior on the streamed chunks - here, we print the streamed
		// responses from the LLM client into the console using the onDelta callback function argument, using an anonymous function.

		turn := history // Remember that GO does cloning which means any change to history does not affect turn from this point
		if retriever != nil {
			// Get the RAG context from topK hits for the last message in history as a single string contextText
			// This only works if the last message in history has a "user" role. Any other role for the last message in
			// history will mean the history is kept as it is.
			contexText, retErr := retriever.Retrieve(ctx, history)
			if retErr != nil {
				fmt.Fprintln(os.Stderr, "retrieval error: ", retErr)
			} else if contexText != "" {
				// Build a turn with the inline context - inject RAG context string into the last message in history
				// if it has been verified to be a user message
				turn = withInlineContext(history, contexText)
			}
		}

		// DEBUG: To view the RAG context injected into the user query
		// if len(turn) > 0 {
		// 	fmt.Fprintf(os.Stderr, "\nFinal Prompt:\n\n%s", turn[len(turn)-1].Content)
		// }

		reply, err := client.ChatStream(ctx, turn, func(s string) {
			// Remember that turn is just a clone of history with additional RAG context in its last message if the last
			// message was having role "user"
			stopOnce.Do(spin.Stop) // Stop the spinner only once, even if this callback is called multiple times.
			fmt.Print(s)           // Print the streamed chat response to the console as it arrives from the LLM.
		})

		stopOnce.Do(spin.Stop) // Ensure the spinner is stopped after the chat response is received.
		fmt.Println()          // print a blank line

		if err != nil {
			fmt.Fprintln(os.Stderr, "error: ", err) // Print the error message to the standard error stream
			history = history[:len(history)-1]      // Remove the last message (user's input) from the conversation history
			continue                                // Continue the REPL loop even when error is encountered from the chat client
		}

		// Update the conversation history with the LLM's reply
		history = append(history, reply)
	}
}

// Function to add a context text to the last message in chat history if the last message in history a message with role "user"
func withInlineContext(history []llm.Message, contexText string) []llm.Message {
	if len(history) == 0 || contexText == "" {
		return history
	}
	last := history[len(history)-1]
	if last.Role != "user" {
		return history
	}

	out := make([]llm.Message, len(history))
	copy(out, history)
	out[len(out)-1] = llm.Message{
		Role:    "user",
		Content: contexText + "\n\n--- Question ---\n\n" + last.Content,
	}
	return out
}

// spinner represents a simple console spinner to indicate that the program is processing.
type spinner struct {
	stop chan struct{} // channel to signal the spinner to stop.
	done chan struct{} // channel to signal that the spinner has stopped.
	once sync.Once
}

// Function to start a spinner with a given label. Returns a pointer to the spinner instance.
func startSpinner(label string) *spinner {
	s := &spinner{
		stop: make(chan struct{}), // in the channel, empty struct{} is used as element with no data, saving memory.
		done: make(chan struct{}),
	}
	go func() {
		defer close(s.done) // close the done channel when this goroutine exits. This signals that the spinner has stopped.
		// Frames for spinner text animation
		frames := []string{"|", "/", "-", "\\"}
		t := time.NewTicker(80 * time.Millisecond) // frame changes once every 80 milliseconds, when this ticker
		// ticks (sends the current time on its channel t.C), the select statement below will handle it.
		defer t.Stop() // release the ticker resources when this function stops

		i := 0
		for {
			select {
			case <-s.stop:
				fmt.Print("\r\033[K") // Clear the spinner line when a stop channel element is received. The carriage return
				// "\r" moves the cursor to the start of the line and the "\033[K" is an ANSI escape sequence that clears
				// the contents in the current line, removing the spinner with its label.
				return
			case <-t.C:
				fmt.Printf("\r%s %s", frames[i%len(frames)], label)
				i++
			}
		}
	}() // goroutine for creating a spinner with label in console is called
	return s
}

// Function to stop the spinner. Ensures that the spinner is stopped only once, even if called multiple times.
func (s *spinner) Stop() {
	s.once.Do(func() { close(s.stop) })
	<-s.done // Wait for the spinner goroutine to signal that it has stopped
}

// function to seed the chat history with messages from a system prompt file.
func seedHistory(path string) ([]llm.Message, error) {
	if path == "" {
		return nil, nil // No system prompt file provided
	}
	data, err := os.ReadFile(path)
	if errors.Is(err, fs.ErrNotExist) {
		return nil, nil // File does not exist, treat as no system prompt
	}
	// return error if unable to read file.
	if err != nil {
		return nil, fmt.Errorf("Read system prompt file: %w", err)
	}

	content := strings.TrimSpace(string(data)) // Remove leading and trailing whitespace
	if content == "" {
		return nil, nil // No content in the system prompt file, treat as no system prompt
	}

	// Return a slice containing a single system message.
	return []llm.Message{
		{
			Role:    "system",
			Content: content,
		},
	}, nil
}
