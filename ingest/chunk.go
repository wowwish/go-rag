package ingest

import "strings"

func chunk(text string, size, overlap int) []string {
	text = strings.TrimSpace(text) // Trim leading and trailing whitespaces from input string
	if text == "" {
		return nil
	}

	// just return the input string as it is, if it is smaller than the size parameter
	if len(text) <= size {
		return []string{text}
	}

	// Overlap is how much each string chunk can have in common with its neighbor chunks from the input string
	// Overlap must be > 0 and must be less than the total size of the input string
	if overlap < 0 {
		overlap = 0
	}

	if overlap >= size {
		overlap = size / 2
	}

	// We want our chunks to be as close to size as possible. If some sentence break occurs in the last 30% length
	// of our chunk, we end the chunk before the break to maintain semantic coherence, clarity and grammatical sanctity
	// of the chunk. Having a chunk with parts of two sentences is not good for RAG and can be picked up as false positive
	//  during semantic searching and can affect performance of the RAG system.
	threshold := size * 7 / 10

	var chunks []string // to hold chunks from the input string with given overlap
	n := len(text)
	start := 0
	// loop through the input string until start reaches its length
	for start < n {
		// end index is start + size, to produce a chunk with length as size
		end := start + size
		// if end becomes higher than the length of input string, just add whatever is the terminal
		// part of the input string to our slice of chunks. This last chunk will be smaller than size.
		if end >= n {
			if part := strings.TrimSpace(text[start:]); part != "" {
				chunks = append(chunks, part)
			}
			break // End the for loop after returning the last chunk
		}

		// Handle current iteration's chunk. Perform Sentence boundary-aware chunking to preserve semantic coherence
		// and clarity (here, we  use a threshold of 30% to lookup sentence breaks in a chunk).
		window := text[start:end]
		switch {
		case strings.LastIndex(window, "\n\n") >= threshold:
			end = start + strings.LastIndex(window, "\n\n") + 2 // substring is 2 characters long
		case strings.LastIndex(window, ". ") >= threshold:
			end = start + strings.LastIndex(window, ". ") + 2 // substring is two characters long
		case strings.LastIndex(window, " ") >= threshold:
			end = start + strings.LastIndex(window, " ") + 1 // substring is one character long
		}

		if part := strings.TrimSpace(text[start:end]); part != "" {
			chunks = append(chunks, part)
		}
		start = end - overlap // Update start for the next iteration
	}

	return chunks
}
