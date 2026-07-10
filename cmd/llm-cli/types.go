package main

import (
	"context"
	"io"
	"time"

	llm "github.com/pkieltyka/go-llm"
)

type app struct {
	stdin           io.Reader
	stdout          io.Writer
	stderr          io.Writer
	providerFactory providerFactory
}

type providerFactory func(context.Context, providerConfig) (llm.Provider, error)

type providerConfig struct {
	name     string
	apiKey   string
	authFile string
	baseURL  string
	timeout  time.Duration
	debug    bool
	stderr   io.Writer
}

type chatConfig struct {
	provider    string
	model       string
	system      string
	effort      llm.Effort
	maxTokens   int
	temperature optionalFloat
	imagePaths  repeatableString
	filePaths   repeatableString
	schemaPath  string
	toolPaths   repeatableString
	noStream    bool
	jsonOutput  bool
	usage       bool
	reasoning   bool
	debug       bool
	cacheSystem bool
	sessionID   string
	loadPath    string
	savePath    string
	apiKey      string
	authFile    string
	baseURL     string
	timeout     time.Duration
	version     bool
	args        []string
	stdinText   string
}

type modelsConfig struct {
	provider   string
	jsonOutput bool
	apiKey     string
	authFile   string
	baseURL    string
	timeout    time.Duration
	debug      bool
	version    bool
}

type requestBundle struct {
	request     *llm.Request
	loaded      []llm.Message
	userMessage *llm.Message
}

type repeatableString []string

func (v *repeatableString) String() string {
	if v == nil {
		return ""
	}
	return ""
}

func (v *repeatableString) Set(s string) error {
	*v = append(*v, s)
	return nil
}

type optionalFloat struct {
	value float64
	set   bool
}

func (v *optionalFloat) String() string {
	if v == nil || !v.set {
		return ""
	}
	return formatFloat(v.value)
}

func (v *optionalFloat) Set(s string) error {
	f, err := parseFloat(s)
	if err != nil {
		return err
	}
	v.value = f
	v.set = true
	return nil
}
