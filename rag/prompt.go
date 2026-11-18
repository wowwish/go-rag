package rag

import (
	"fmt"
	"rag-course/vector"
	"strings"
)

// Package retrieved chunks into the text block for our chat

const contextPreamble = `Use the following excerpts from the document collection to answer question.
Cite sources by filename when you draw from them. If the excerpts do not address the question, say so
before answering from general knowledge.`

const unknownSource = "{unknown source}" // Used for information from unknown sources

// function to package retrieved chunks from vector store into a single text block for pre-pending with user query
func formatContext(hits []vector.Result) string {
	if len(hits) == 0 {
		return ""
	}

	var sb strings.Builder
	sb.WriteString(contextPreamble)
	sb.WriteString("\n\n-----Excerpts-----\n\n")

	for i, h := range hits {
		source := h.Metadata["source"]
		if source == "" {
			source = unknownSource
		}
		fmt.Fprintf(&sb, "[%d] Source: %s (similarity %.2f)\n%s\n\n", i+1, source, h.Score, h.Content)
	}

	return strings.TrimSpace(sb.String())
}
