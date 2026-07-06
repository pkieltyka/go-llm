package main

import (
	"encoding/json"
	"fmt"
	"mime"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strings"

	llm "github.com/pkieltyka/go-llm"
)

func buildRequest(cfg chatConfig) (requestBundle, error) {
	var bundle requestBundle
	if strings.TrimSpace(cfg.model) == "" {
		return bundle, fmt.Errorf("%w: missing model; set -m/--model", llm.ErrBadRequest)
	}

	if cfg.loadPath != "" {
		data, err := os.ReadFile(cfg.loadPath)
		if err != nil {
			return bundle, fmt.Errorf("load conversation: %w", err)
		}
		msgs, err := llm.UnmarshalMessages(data)
		if err != nil {
			return bundle, fmt.Errorf("load conversation: %w", err)
		}
		bundle.loaded = msgs
	}

	userParts, err := buildUserParts(cfg)
	if err != nil {
		return bundle, err
	}
	messages := append([]llm.Message(nil), bundle.loaded...)
	if len(userParts) > 0 {
		msg := llm.UserParts(userParts...)
		bundle.userMessage = &msg
		messages = append(messages, msg)
	}
	if len(messages) == 0 {
		return bundle, fmt.Errorf("%w: provide a prompt, stdin, attachment, or --load conversation", llm.ErrBadRequest)
	}

	req := &llm.Request{
		Model:          cfg.model,
		Messages:       messages,
		System:         cfg.system,
		MaxTokens:      cfg.maxTokens,
		Effort:         cfg.effort,
		SessionID:      cfg.sessionID,
		ResponseFormat: nil,
	}
	if cfg.temperature.set {
		temp := cfg.temperature.value
		req.Temperature = &temp
	}
	if cfg.cacheSystem {
		req.SystemCache = &llm.CacheHint{}
	}
	if cfg.schemaPath != "" {
		format, err := loadResponseFormat(cfg.schemaPath)
		if err != nil {
			return bundle, err
		}
		req.ResponseFormat = format
	}
	tools, err := loadTools(cfg.toolPaths)
	if err != nil {
		return bundle, err
	}
	req.Tools = tools

	bundle.request = req
	return bundle, nil
}

func buildUserParts(cfg chatConfig) ([]llm.Part, error) {
	var parts []llm.Part
	if prompt := promptFromArgs(cfg.args); prompt != "" {
		parts = append(parts, llm.Text(prompt))
	}
	if stdin := strings.TrimRight(cfg.stdinText, "\n"); stdin != "" {
		parts = append(parts, llm.Text(stdin))
	}
	for _, path := range cfg.imagePaths {
		part, err := loadImagePart(path)
		if err != nil {
			return nil, err
		}
		parts = append(parts, part)
	}
	for _, path := range cfg.filePaths {
		part, err := loadFilePart(path)
		if err != nil {
			return nil, err
		}
		parts = append(parts, part)
	}
	return parts, nil
}

func loadImagePart(value string) (llm.Part, error) {
	if isURL(value) {
		return llm.ImageURL(value), nil
	}
	data, err := os.ReadFile(value)
	if err != nil {
		return nil, fmt.Errorf("read image %q: %w", value, err)
	}
	return llm.ImageData(data, mediaType(value, data)), nil
}

func loadFilePart(value string) (llm.Part, error) {
	if isURL(value) {
		return llm.FileURL(value, mediaType(value, nil)), nil
	}
	data, err := os.ReadFile(value)
	if err != nil {
		return nil, fmt.Errorf("read file %q: %w", value, err)
	}
	return llm.FileData(data, mediaType(value, data), filepath.Base(value)), nil
}

func isURL(value string) bool {
	u, err := url.Parse(value)
	return err == nil && (u.Scheme == "http" || u.Scheme == "https") && u.Host != ""
}

func mediaType(path string, data []byte) string {
	if ext := filepath.Ext(path); ext != "" {
		if mt := mime.TypeByExtension(ext); mt != "" {
			return mt
		}
	}
	if len(data) > 0 {
		return http.DetectContentType(data)
	}
	return ""
}

func loadResponseFormat(path string) (*llm.ResponseFormat, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read schema %q: %w", path, err)
	}
	raw := json.RawMessage(append([]byte(nil), data...))
	if !json.Valid(raw) {
		return nil, fmt.Errorf("%w: schema %q is not valid JSON", llm.ErrBadRequest, path)
	}
	name := strings.TrimSuffix(filepath.Base(path), filepath.Ext(path))
	if name == "" {
		name = "response"
	}
	return &llm.ResponseFormat{
		Type:   llm.FormatJSONSchema,
		Name:   name,
		Schema: raw,
		Strict: true,
	}, nil
}

func loadTools(paths []string) ([]llm.Tool, error) {
	var tools []llm.Tool
	for _, path := range paths {
		loaded, err := loadToolFile(path)
		if err != nil {
			return nil, err
		}
		tools = append(tools, loaded...)
	}
	return tools, nil
}

func loadToolFile(path string) ([]llm.Tool, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read tool %q: %w", path, err)
	}
	var list []toolFile
	if err := json.Unmarshal(data, &list); err == nil {
		return toolsFromFile(path, list)
	}
	var one toolFile
	if err := json.Unmarshal(data, &one); err != nil {
		return nil, fmt.Errorf("parse tool %q: %w", path, err)
	}
	return toolsFromFile(path, []toolFile{one})
}

type toolFile struct {
	Name        string          `json:"name"`
	Description string          `json:"description"`
	InputSchema json.RawMessage `json:"input_schema"`
	Schema      json.RawMessage `json:"schema"`
	Strict      bool            `json:"strict"`
}

func toolsFromFile(path string, files []toolFile) ([]llm.Tool, error) {
	out := make([]llm.Tool, 0, len(files))
	for _, file := range files {
		if strings.TrimSpace(file.Name) == "" {
			return nil, fmt.Errorf("%w: tool %q missing name", llm.ErrBadRequest, path)
		}
		schema := file.InputSchema
		if len(schema) == 0 {
			schema = file.Schema
		}
		if len(schema) == 0 {
			schema = json.RawMessage(`{"type":"object","properties":{}}`)
		}
		if !json.Valid(schema) {
			return nil, fmt.Errorf("%w: tool %q schema is not valid JSON", llm.ErrBadRequest, file.Name)
		}
		out = append(out, llm.Tool{
			Name:        file.Name,
			Description: file.Description,
			InputSchema: json.RawMessage(append([]byte(nil), schema...)),
			Strict:      file.Strict,
		})
	}
	return out, nil
}

func historyMessages(bundle requestBundle, resp *llm.Response) []llm.Message {
	h := llm.NewHistory()
	h.Add(bundle.loaded...)
	if bundle.userMessage != nil {
		h.Add(*bundle.userMessage)
	}
	h.AddResponse(resp)
	return h.Messages()
}
