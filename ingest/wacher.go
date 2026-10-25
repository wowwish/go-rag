package ingest

import (
	"context"
	"errors"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"rag-course/llm"
	"rag-course/vector"
	"strings"
	"sync"
	"time"

	"github.com/fsnotify/fsnotify"
)

// Delay to allow file copying processes to complete before the detected new file will start processing
const debounceDelay = 500 * time.Millisecond

func Watch(ctx context.Context, opts Options, embedder llm.Embedder, store vector.Store, logger *log.Logger) error {
	if filepath.Clean(opts.SourceDir) == filepath.Clean(opts.ProcessedDir) {
		return errors.New("source and process directories must be different")
	}

	// Make all directories in the given path recursively if they donot already exist, with the 755 permission flag for read,write,execute
	// permissions
	if err := os.MkdirAll(opts.SourceDir, 0o755); err != nil {
		return fmt.Errorf("create source dir: %w", err)
	}
	if err := os.MkdirAll(opts.ProcessedDir, 0o755); err != nil {
		return fmt.Errorf("create process dir: %w", err)
	}

	// Create a watcher
	w, err := fsnotify.NewWatcher()
	if err != nil {
		return fmt.Errorf("create watcher: %w", err)
	}
	defer w.Close()

	// Add the source dir to the watcher
	if err := w.Add(opts.SourceDir); err != nil {
		return fmt.Errorf("watch source dir: %w", err)
	}

	// Get the absolute directory path for process dir
	processAbs, err := filepath.Abs(opts.ProcessedDir)
	if err != nil {
		return fmt.Errorf("resolve processed dir: %w", err)
	}

	// Setup an anonymous function to handle file processing on the source dir - we use processOne function that was declared
	// in ingest.go script here
	handle := func(path string) {
		if err := processOne(ctx, path, opts, embedder, store); err != nil {
			logger.Printf("process %s: %v", filepath.Base(path), err)
			return // exit the anonymous function
		}
		dst := filepath.Join(opts.ProcessedDir, filepath.Base(path)) // Build the destination file to move processed source file
		// os.Rename here is used to move a processed source file to its destination file in processed dir
		if err := os.Rename(path, dst); err != nil {
			logger.Printf("move %s to processed: %v", err)
			return // exit the anonymous function
		}
		logger.Printf("ingested %s", filepath.Base(path))
	}

	// Handle the case where there are already files in the source dir when starting this program:
	// Read files and directories from source dir
	entries, err := os.ReadDir(opts.SourceDir)
	if err != nil {
		return fmt.Errorf("read source dir: %w", err)
	}
	for _, e := range entries {
		// skip directories
		if e.IsDir() || strings.HasPrefix(e.Name(), ".") {
			continue
		}
		go handle(filepath.Join(opts.SourceDir, e.Name())) // process validated files from the source dir in the background
		// using a goroutine
	}

	// Handle the case where files are dropped into the source dir while running this program:
	var (
		timersMu sync.Mutex                     // A mutex lock
		timers   = make(map[string]*time.Timer) // The time.Timer type represents a single event. When the timer expires,
		// the current time will be sent on the type's internal Channel Timer.C
		// Timer objects can be created using the time.NewTimer() and time.AfterFunc() methods.
		// The timers created using time.NewTimer() will simply return current time to its internal channel Timer.C after the delay.
		// timers created using Timer.AfterFunc() will run a function after an initial delay. The triggered function by the timer
		// may already be already running from a different call and the timer wouldn't know it, hence, it can also concurrently
		// execute the same function. So, it is neccessary to handle this problem to prevent data corruption by using mutex
		// locks to allow only one function instance access, as we do here for the timers map.
	)

	// An anonymous function to wait for a period of time (debounceDelay) before processing a file
	schedule := func(path string) {
		timersMu.Lock()         // Lock the timers map while accessing it
		defer timersMu.Unlock() // Unlock the mutex and allow access to the timers map when exiting this anonymous function
		// If the a timer already exists for the given path in our map (meaning the file was altered), pick up the timer
		// object specific to that file and make it active for the delay duration.
		if t, ok := timers[path]; ok {
			// Reset the timer to run the file processing function after the configured delay to give some time for the write of
			// the newly detected file in source dir to complete. After the time delay from detection, we can start
			// processing the file without interruption or corrupted data.
			t.Reset(debounceDelay) // This will cause the timer object to call a processing function after an initial delay of
			// debounceDelay.
			return
		}

		// wait for debounceDelay time and then
		timers[path] = time.AfterFunc(debounceDelay, func() {
			timersMu.Lock()
			delete(timers, path)
			timersMu.Unlock()
			handle(path)
		})
	}

	for {
		select {
		case <-ctx.Done():
			return nil
		case ev, ok := <-w.Events:
			if !ok {
				return nil
			}
			if ev.Op&(fsnotify.Create|fsnotify.Write) == 0 {
				continue
			}
			if !shouldProcess(ev.Name, processAbs) {
				continue
			}
		}
	}
}
