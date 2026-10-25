package main

import (
	"context" // Import the context package for managing cancellation and timeouts
	"fmt"
	"os"        // Import the os package for interacting with the operating system
	"os/signal" // Import the os/signal package for handling incoming signals
	"rag-course/app"
	"rag-course/config"
	"syscall" // Import the syscall package for low-level system calls
)

func main() {
	// Create a context that is canceled when an interrupt or termination signal is received. This helps with
	// graceful shutdown of the application, cleaning up memory and other resources as needed. The same context
	// reaches the REPL as well as the HTTP request to the chat endpoint.
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop() // Ensure the context is canceled and resources are cleaned up on exit from this main function

	// Set up the App. Calling app.Run will actually call chat.runREPL, which runs the Read Evaluate Print Loop (REPL)
	// for interactive user input and response generation. Later, it will also run the web based version.
	if err := app.Run(ctx, config.Load()); err != nil {
		fmt.Fprintln(os.Stderr, err) // Print the error to the standard error output if the app fails to run
		os.Exit(1)                   // Exit the program with a non-zero status code to indicate an error occurred
	}

	// Set up config

	// Set up LLM client

	// Set up the Read Evaluate Print Loop (REPL)

}
