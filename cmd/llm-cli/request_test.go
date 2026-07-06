package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	llm "github.com/pkieltyka/go-llm"
)

func TestBuildRequestFromFlags(t *testing.T) {
	dir := t.TempDir()
	imagePath := filepath.Join(dir, "image.png")
	filePath := filepath.Join(dir, "doc.pdf")
	if err := os.WriteFile(imagePath, []byte("png-ish"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filePath, []byte("%PDF-ish"), 0o600); err != nil {
		t.Fatal(err)
	}

	cfg := chatConfig{
		model:       "model-1",
		system:      "system",
		effort:      llm.EffortHigh,
		maxTokens:   123,
		temperature: optionalFloat{value: 0.7, set: true},
		cacheSystem: true,
		sessionID:   "session-1",
		args:        []string{"hello", "there"},
		stdinText:   "stdin text\n",
		imagePaths:  repeatableString{imagePath, "https://example.test/image.jpg"},
		filePaths:   repeatableString{filePath, "https://example.test/file.pdf"},
	}
	bundle, err := buildRequest(cfg)
	if err != nil {
		t.Fatal(err)
	}
	req := bundle.request
	if req.Model != "model-1" || req.System != "system" || req.Effort != llm.EffortHigh {
		t.Fatalf("unexpected scalar fields: %+v", req)
	}
	if req.MaxTokens != 123 || req.SessionID != "session-1" {
		t.Fatalf("unexpected budget/session fields: %+v", req)
	}
	if req.Temperature == nil || *req.Temperature != 0.7 {
		t.Fatalf("temperature not set: %+v", req.Temperature)
	}
	if req.SystemCache == nil {
		t.Fatal("SystemCache was not set")
	}
	if len(req.Messages) != 1 || req.Messages[0].Role != llm.RoleUser {
		t.Fatalf("unexpected messages: %+v", req.Messages)
	}
	parts := req.Messages[0].Parts
	if len(parts) != 6 {
		t.Fatalf("parts len = %d, want 6", len(parts))
	}
	if got := parts[0].(llm.TextPart).Text; got != "hello there" {
		t.Fatalf("prompt text = %q", got)
	}
	if got := parts[1].(llm.TextPart).Text; got != "stdin text" {
		t.Fatalf("stdin text = %q", got)
	}
	if got := parts[2].(llm.ImagePart).MediaType; got != "image/png" {
		t.Fatalf("image media type = %q", got)
	}
	if got := parts[3].(llm.ImagePart).URL; got != "https://example.test/image.jpg" {
		t.Fatalf("image URL = %q", got)
	}
	if got := parts[4].(llm.FilePart).Name; got != "doc.pdf" {
		t.Fatalf("file name = %q", got)
	}
	if got := parts[5].(llm.FilePart).URL; got != "https://example.test/file.pdf" {
		t.Fatalf("file URL = %q", got)
	}
}

func TestBuildRequestSchemaAndTools(t *testing.T) {
	dir := t.TempDir()
	schemaPath := filepath.Join(dir, "answer.json")
	toolPath := filepath.Join(dir, "tool.json")
	if err := os.WriteFile(schemaPath, []byte(`{"type":"object","properties":{"answer":{"type":"string"}}}`), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(toolPath, []byte(`{"name":"lookup","description":"Lookup data","input_schema":{"type":"object","properties":{"q":{"type":"string"}}},"strict":true}`), 0o600); err != nil {
		t.Fatal(err)
	}

	bundle, err := buildRequest(chatConfig{
		model:      "model-1",
		args:       []string{"hi"},
		schemaPath: schemaPath,
		toolPaths:  repeatableString{toolPath},
	})
	if err != nil {
		t.Fatal(err)
	}
	req := bundle.request
	if req.ResponseFormat == nil || req.ResponseFormat.Type != llm.FormatJSONSchema || req.ResponseFormat.Name != "answer" || !req.ResponseFormat.Strict {
		t.Fatalf("unexpected response format: %+v", req.ResponseFormat)
	}
	if len(req.Tools) != 1 {
		t.Fatalf("tools len = %d, want 1", len(req.Tools))
	}
	tool := req.Tools[0]
	if tool.Name != "lookup" || tool.Description != "Lookup data" || !tool.Strict {
		t.Fatalf("unexpected tool: %+v", tool)
	}
	raw, ok := tool.InputSchema.(json.RawMessage)
	if !ok || !json.Valid(raw) {
		t.Fatalf("tool schema is not raw valid JSON: %T %s", tool.InputSchema, raw)
	}
}
