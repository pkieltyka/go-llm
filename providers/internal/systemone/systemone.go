// Package systemone implements TypeSafe's System One classification
// protocol, which TypeSafe, OpenRouter, Vercel AI Gateway, OpenCode Zen,
// llama.cpp and (inside its own envelope) Cloudflare Workers AI serve
// (FS §13C).
//
// A request is POST <base>/systemone with {model, state, questions,
// assets?, images?}; the public bool question is "noul" on the wire. The
// response is {model?, id?, answers, usage: {input_tokens, output_tokens,
// cost?}}. Images take one of three wire forms (ImageForm).
package systemone

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"strconv"
	"strings"
	"time"

	llm "github.com/pkieltyka/go-llm"
	"github.com/pkieltyka/go-llm/providers/internal/providerutil"
)

// ImageForm is how a service takes images.
type ImageForm int

const (
	// ImageAssets is TypeSafe's assets map: {"<image-1>": {type: "image",
	// format: "png"|"jpeg", data_base64}}, with the names placed before the
	// state, state: ["<image-1>", ..., state]. PNG and JPEG only.
	ImageAssets ImageForm = iota
	// ImageDataURLs is an images list of data URLs beside the unchanged
	// state: Clef's form, and llama.cpp's (its images/files). PNG, JPEG and
	// WebP; a service's own limits belong in its Preflight.
	ImageDataURLs
	// ImageStateURLs is OpenRouter's form: image_url parts placed before the
	// state, state: [{type: "image_url", image_url: {url}}, ..., state].
	ImageStateURLs
)

// imageOptInHeader is TypeSafe's opt-in header for its experimental image
// input; without it TypeSafe's own API answers 400.
const imageOptInHeader = "TypeSafe-Danger-Zone-Images"

// Config describes one System One service.
type Config struct {
	// Provider names the service in results, errors and logs.
	Provider string
	// BaseURL is the service root; requests go to <BaseURL>/systemone
	// unless Endpoint is set.
	BaseURL string
	// Endpoint, when set, resolves the request URL and the bearer key
	// together, once per request (Cloudflare, whose account and key come
	// from one credential read); APIKey is then unused.
	Endpoint func(context.Context) (url, key string, err error)
	// Payload, when set, wraps the System One request in the service's
	// envelope (Cloudflare's {model, input}); the default sends the request
	// with its model.
	Payload func(model string, request map[string]any) any
	// Output, when set, extracts the System One output from the service's
	// response envelope; the default reads the body as the output.
	Output func(status int, body []byte) (json.RawMessage, error)
	// APIKey returns the bearer key; nil or "" sends no Authorization.
	APIKey func(context.Context) (string, error)
	// Headers are sent with every request.
	Headers http.Header
	// ImageForm returns the image form for a model; nil means ImageAssets.
	ImageForm func(model string) ImageForm
	// ImageOptIn says whether a request with images sends TypeSafe's
	// opt-in header.
	ImageOptIn bool
	// TextOnly rejects images before I/O (services that list their
	// classifier models as text-only).
	TextOnly bool
	// Preflight, when set, checks the service's documented limits before
	// I/O, after the common validation.
	Preflight func(*llm.ClassifyRequest) error
	// ErrorKind, when set, classifies the service's native error codes
	// before the shared table.
	ErrorKind providerutil.ErrorKindHook

	HTTPClient      *http.Client
	MaxRetries      int
	ResponseRetries bool
	Timeout         time.Duration
	PriceTable      llm.PriceTable
	Logger          *slog.Logger
	WireCapture     func(llm.WireCapture)
}

// Engine classifies through one System One service.
type Engine struct {
	cfg    Config
	client *http.Client
}

// New builds an engine. The HTTP client is wrapped for logging, wire
// capture and safe retries like the chat engines' clients.
func New(cfg Config) (*Engine, error) {
	if cfg.Provider == "" {
		return nil, fmt.Errorf("%w: system one provider needs a name", llm.ErrBadRequest)
	}
	if cfg.Endpoint == nil && cfg.BaseURL == "" {
		return nil, fmt.Errorf("%w: %s needs a base URL", llm.ErrBadRequest, cfg.Provider)
	}
	if cfg.MaxRetries < 0 {
		return nil, fmt.Errorf("%w: max retries must be >= 0", llm.ErrBadRequest)
	}
	observed := providerutil.ObservedHTTPClient(cfg.HTTPClient, cfg.Provider, cfg.Logger, cfg.WireCapture)
	return &Engine{cfg: cfg, client: providerutil.SafeRetryHTTPClient(observed, cfg.MaxRetries, cfg.ResponseRetries)}, nil
}

// Classify sends req and parses the answers. A billed response with
// malformed answers returns the result (with its usage) and an error
// wrapping ErrServer.
func (e *Engine) Classify(ctx context.Context, req *llm.ClassifyRequest) (*llm.ClassifyResult, error) {
	start := time.Now()
	result, err := e.classify(ctx, req)
	if err != nil {
		e.logFailure(ctx, req, start, err)
		return result, err
	}
	if e.cfg.Logger != nil {
		e.cfg.Logger.DebugContext(ctx, "llm provider classify",
			slog.String("provider", e.cfg.Provider),
			slog.String("model", result.Model),
			slog.Duration("duration", time.Since(start)),
			slog.Int("questions", len(req.Questions)),
			slog.Int64("input_tokens", result.Usage.InputTokens),
			slog.Int64("output_tokens", result.Usage.OutputTokens),
		)
	}
	return result, nil
}

func (e *Engine) classify(ctx context.Context, req *llm.ClassifyRequest) (*llm.ClassifyResult, error) {
	if err := llm.ValidateClassifyRequest(req); err != nil {
		return nil, err
	}
	if e.cfg.Preflight != nil {
		if err := e.cfg.Preflight(req); err != nil {
			return nil, err
		}
	}
	request, optIn, err := e.wireRequest(req)
	if err != nil {
		return nil, err
	}
	var payload any = request
	if e.cfg.Payload != nil {
		payload = e.cfg.Payload(req.Model, request)
	} else {
		request["model"] = req.Model
	}
	body, err := marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("%w: encode %s classify request: %v", llm.ErrBadRequest, e.cfg.Provider, err)
	}
	if e.cfg.Timeout > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, e.cfg.Timeout)
		defer cancel()
	}
	status, respBody, err := e.post(ctx, body, optIn)
	if err != nil {
		return nil, err
	}
	output := json.RawMessage(respBody)
	if e.cfg.Output != nil {
		if output, err = e.cfg.Output(status, respBody); err != nil {
			return nil, err
		}
	}
	return ParseOutput(e.cfg.Provider, req, output, e.cfg.PriceTable)
}

func (e *Engine) post(ctx context.Context, body []byte, optIn bool) (int, []byte, error) {
	url := strings.TrimRight(e.cfg.BaseURL, "/") + "/systemone"
	var key string
	switch {
	case e.cfg.Endpoint != nil:
		var err error
		if url, key, err = e.cfg.Endpoint(ctx); err != nil {
			return 0, nil, err
		}
	case e.cfg.APIKey != nil:
		var err error
		if key, err = e.cfg.APIKey(ctx); err != nil {
			return 0, nil, err
		}
	}
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return 0, nil, fmt.Errorf("%w: %s classify request: %v", llm.ErrBadRequest, e.cfg.Provider, err)
	}
	httpReq.GetBody = func() (io.ReadCloser, error) { return io.NopCloser(bytes.NewReader(body)), nil }
	for key, values := range e.cfg.Headers {
		for _, value := range values {
			httpReq.Header.Add(key, value)
		}
	}
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Accept", "application/json")
	if optIn {
		httpReq.Header.Set(imageOptInHeader, "true")
	}
	if key != "" {
		httpReq.Header.Set("Authorization", "Bearer "+key)
	}
	resp, err := e.client.Do(httpReq)
	if err != nil {
		return 0, nil, providerutil.NormalizeRemoteError(e.cfg.Provider, err)
	}
	defer resp.Body.Close()
	respBody, err := io.ReadAll(io.LimitReader(resp.Body, 16<<20))
	if err != nil {
		return 0, nil, providerutil.NormalizeRemoteError(e.cfg.Provider, err)
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return resp.StatusCode, nil, HTTPError(e.cfg.Provider, resp, respBody, e.cfg.ErrorKind)
	}
	return resp.StatusCode, respBody, nil
}

// HTTPError builds the normalized error for a non-2xx response. It reads
// the {"error": {"message", "code"}} shape, a bare {"message", "code"} and
// FastAPI's {"detail": ...}, and falls back to the body text. A non-nil
// hook classifies the service's native codes first.
func HTTPError(provider string, resp *http.Response, body []byte, hook providerutil.ErrorKindHook) error {
	code, message := parseErrorBody(body)
	if message == "" {
		message = http.StatusText(resp.StatusCode)
	}
	return &llm.ProviderError{
		Provider:   provider,
		HTTPStatus: resp.StatusCode,
		Code:       code,
		Message:    message,
		RetryAfter: llm.RetryAfter(resp),
		RawBody:    body,
		Kind:       ErrorKind(resp.StatusCode, code, message, hook),
	}
}

func parseErrorBody(body []byte) (string, string) {
	var wrapped struct {
		Error struct {
			Message string `json:"message"`
			Code    any    `json:"code"`
			Type    string `json:"type"`
		} `json:"error"`
		Message   string          `json:"message"`
		Code      any             `json:"code"`
		ErrorType string          `json:"error_type"`
		Detail    json.RawMessage `json:"detail"`
		Errors    []struct {
			Code    any    `json:"code"`
			Message string `json:"message"`
		} `json:"errors"`
	}
	if json.Unmarshal(body, &wrapped) == nil {
		switch {
		case len(wrapped.Errors) > 0 && wrapped.Errors[0].Message != "":
			// Cloudflare: {"success": false, "errors": [{"code", "message"}]}.
			messages := make([]string, 0, len(wrapped.Errors))
			for _, e := range wrapped.Errors {
				if e.Message != "" {
					messages = append(messages, e.Message)
				}
			}
			return CodeString(wrapped.Errors[0].Code), strings.Join(messages, "; ")
		case wrapped.Error.Message != "":
			code := CodeString(wrapped.Error.Code)
			if code == "" {
				code = wrapped.Error.Type
			}
			return code, wrapped.Error.Message
		case wrapped.Message != "":
			code := CodeString(wrapped.Code)
			if code == "" {
				code = wrapped.ErrorType
			}
			return code, wrapped.Message
		case len(wrapped.Detail) > 0:
			// FastAPI: {"detail": "text"} or TypeSafe's {"detail":
			// {"error_type", "message"}}.
			var text string
			if json.Unmarshal(wrapped.Detail, &text) == nil {
				return "", text
			}
			var detail struct {
				ErrorType string `json:"error_type"`
				Message   string `json:"message"`
			}
			if json.Unmarshal(wrapped.Detail, &detail) == nil && detail.Message != "" {
				return detail.ErrorType, detail.Message
			}
			return "", string(wrapped.Detail)
		}
	}
	return "", strings.TrimSpace(string(body))
}

// ErrorKind classifies a System One error: the service's hook, then
// TypeSafe's exact codes, then the shared table.
func ErrorKind(status int, code, message string, hook providerutil.ErrorKindHook) error {
	hooks := []providerutil.ErrorKindHook{typeSafeHook}
	if hook != nil {
		hooks = append([]providerutil.ErrorKindHook{hook}, hooks...)
	}
	return providerutil.ErrorKind(status, code, message, hooks...)
}

// typeSafeHook classifies TypeSafe's authentication_error type (sent with
// 403) as ErrAuth and its exact max_tokens_exceeded code (the state is too
// long for the model) as ErrContextTooLong.
func typeSafeHook(_ int, code, _ string) error {
	switch code {
	case "authentication_error":
		return llm.ErrAuth
	case "max_tokens_exceeded":
		return llm.ErrContextTooLong
	}
	return nil
}

// CodeString renders a native error code, a JSON string or number, as text.
func CodeString(code any) string {
	switch v := code.(type) {
	case nil:
		return ""
	case string:
		return v
	case float64:
		return strconv.FormatFloat(v, 'f', -1, 64)
	default:
		encoded, _ := json.Marshal(v)
		return string(encoded)
	}
}

func (e *Engine) logFailure(ctx context.Context, req *llm.ClassifyRequest, start time.Time, err error) {
	if e.cfg.Logger == nil {
		return
	}
	model := ""
	if req != nil {
		model = req.Model
	}
	e.cfg.Logger.ErrorContext(ctx, "llm provider classify failed",
		slog.String("provider", e.cfg.Provider),
		slog.String("model", model),
		slog.Duration("duration", time.Since(start)),
		slog.String("error", llm.SafeError(err)),
	)
}

// wireRequest builds the System One request (without its model) and says
// whether it needs the image opt-in header.
func (e *Engine) wireRequest(req *llm.ClassifyRequest) (map[string]any, bool, error) {
	questions := make(orderedObject, 0, len(req.Questions))
	for _, q := range req.Questions {
		questions = append(questions, member{q.Name, WireQuestion(q)})
	}
	request := map[string]any{
		"state":     json.RawMessage(req.State),
		"questions": questions,
	}
	if len(req.Images) == 0 {
		return request, false, nil
	}
	if e.cfg.TextOnly {
		return nil, false, fmt.Errorf("%w: %s classifier models take text only", llm.ErrUnsupported, e.cfg.Provider)
	}
	form := ImageAssets
	if e.cfg.ImageForm != nil {
		form = e.cfg.ImageForm(req.Model)
	}
	if err := addImages(e.cfg.Provider, form, req, request); err != nil {
		return nil, false, err
	}
	return request, e.cfg.ImageOptIn && form == ImageAssets, nil
}

// WireQuestion maps a question to System One's {type, instructions,
// criteria}: bool is "noul" with {"true", "false"} meanings, choice
// criteria an object of value to description, score criteria the levels.
func WireQuestion(q llm.Question) any {
	switch q.Type {
	case llm.QuestionChoice:
		criteria := make(orderedObject, 0, len(q.Choices))
		for _, choice := range q.Choices {
			criteria = append(criteria, member{choice.Value, choice.Description})
		}
		return orderedObject{{"type", "choice"}, {"instructions", q.Instructions}, {"criteria", criteria}}
	case llm.QuestionScore:
		return orderedObject{{"type", "score"}, {"instructions", q.Instructions}, {"criteria", q.Levels}}
	default:
		return orderedObject{{"type", "noul"}, {"instructions", q.Instructions},
			{"criteria", orderedObject{{"true", q.True}, {"false", q.False}}}}
	}
}

func addImages(provider string, form ImageForm, req *llm.ClassifyRequest, request map[string]any) error {
	images := req.Images
	switch form {
	case ImageDataURLs:
		urls := make([]string, len(images))
		for i, image := range images {
			switch image.MediaType {
			case "image/png", "image/jpeg", "image/webp":
			default:
				return fmt.Errorf("%w: %s accepts PNG, JPEG and WebP images, got %s", llm.ErrBadRequest, provider, image.MediaType)
			}
			urls[i] = dataURL(image)
		}
		request["images"] = urls
	case ImageStateURLs:
		state := make([]any, 0, len(images)+1)
		for _, image := range images {
			state = append(state, map[string]any{"type": "image_url", "image_url": map[string]string{"url": dataURL(image)}})
		}
		request["state"] = append(state, request["state"])
	default:
		prefix := assetPrefix(req)
		assets := make(orderedObject, 0, len(images))
		state := make([]any, 0, len(images)+1)
		for i, image := range images {
			var format string
			switch image.MediaType {
			case "image/png":
				format = "png"
			case "image/jpeg":
				format = "jpeg"
			default:
				return fmt.Errorf("%w: %s accepts PNG and JPEG images, got %s", llm.ErrBadRequest, provider, image.MediaType)
			}
			name := prefix + strconv.Itoa(i+1) + ">"
			assets = append(assets, member{name, map[string]string{
				"type": "image", "format": format, "data_base64": base64.StdEncoding.EncodeToString(image.Data),
			}})
			state = append(state, name)
		}
		request["assets"] = assets
		request["state"] = append(state, request["state"])
	}
	return nil
}

// assetPrefix returns the prefix of the asset names, "<image-" unless a
// caller string contains it: TypeSafe substitutes an asset name wherever it
// occurs in the request's strings, so a literal "<image-1>" in the state,
// instructions or criteria must not match one. The fallback prefixes
// ("<image1-", "<image2-", ...) are deterministic.
func assetPrefix(req *llm.ClassifyRequest) string {
	var texts []string
	var walk func(any)
	walk = func(v any) {
		switch v := v.(type) {
		case string:
			texts = append(texts, v)
		case map[string]any:
			for key, value := range v {
				texts = append(texts, key)
				walk(value)
			}
		case []any:
			for _, value := range v {
				walk(value)
			}
		}
	}
	decoder := json.NewDecoder(bytes.NewReader(req.State))
	decoder.UseNumber()
	var state any
	if decoder.Decode(&state) == nil {
		walk(state)
	}
	for _, q := range req.Questions {
		texts = append(texts, q.Name, q.Instructions, q.True, q.False)
		texts = append(texts, q.Levels...)
		for _, c := range q.Choices {
			texts = append(texts, c.Value, c.Description)
		}
	}
	prefix := "<image-"
	for n := 1; ; n++ {
		used := false
		for _, text := range texts {
			if strings.Contains(text, prefix) {
				used = true
				break
			}
		}
		if !used {
			return prefix
		}
		prefix = "<image" + strconv.Itoa(n) + "-"
	}
}

func dataURL(image llm.ImagePart) string {
	return "data:" + image.MediaType + ";base64," + base64.StdEncoding.EncodeToString(image.Data)
}

// orderedObject marshals as a JSON object in its member order, so
// questions and choices reach the model in the order the caller gave.
type orderedObject []member

type member struct {
	key   string
	value any
}

func (o orderedObject) MarshalJSON() ([]byte, error) {
	var buf bytes.Buffer
	buf.WriteByte('{')
	for i, m := range o {
		if i > 0 {
			buf.WriteByte(',')
		}
		key, err := marshal(m.key)
		if err != nil {
			return nil, err
		}
		value, err := marshal(m.value)
		if err != nil {
			return nil, err
		}
		buf.Write(key)
		buf.WriteByte(':')
		buf.Write(value)
	}
	buf.WriteByte('}')
	return buf.Bytes(), nil
}

// marshal encodes v without HTML escaping, so asset names such as
// "<image-1>" go out as written.
func marshal(v any) ([]byte, error) {
	var buf bytes.Buffer
	enc := json.NewEncoder(&buf)
	enc.SetEscapeHTML(false)
	if err := enc.Encode(v); err != nil {
		return nil, err
	}
	return bytes.TrimRight(buf.Bytes(), "\n"), nil
}
