package llmtest

import (
	"context"
	"encoding/json"
	"fmt"
	"iter"
	"sync"

	llm "github.com/pkieltyka/go-llm"
)

// Provider is a goroutine-safe scripted fake provider.
type Provider struct {
	mu           sync.Mutex
	name         string
	capabilities []llm.Capability
	models       []llm.ModelInfo
	steps        []step
	requests     []*llm.Request
}

type stepKind int

const (
	stepResponse stepKind = iota + 1
	stepStream
	stepError
)

type step struct {
	kind   stepKind
	resp   *llm.Response
	events []llm.Event
	err    error
}

// Option configures a fake provider.
type Option func(*Provider)

// New returns a configured fake provider.
func New(opts ...Option) *Provider {
	p := &Provider{name: "llmtest"}
	for _, opt := range opts {
		opt(p)
	}
	return p
}

// WithName sets the provider name.
func WithName(name string) Option {
	return func(p *Provider) { p.name = name }
}

// WithCapabilities sets provider capabilities.
func WithCapabilities(caps ...llm.Capability) Option {
	return func(p *Provider) { p.capabilities = append([]llm.Capability(nil), caps...) }
}

// WithModels sets the model list returned by Models.
func WithModels(models ...llm.ModelInfo) Option {
	return func(p *Provider) { p.models = cloneModels(models) }
}

func (p *Provider) Name() string {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.name
}

func (p *Provider) Capabilities() []llm.Capability {
	p.mu.Lock()
	defer p.mu.Unlock()
	return append([]llm.Capability(nil), p.capabilities...)
}

func (p *Provider) Models(context.Context) ([]llm.ModelInfo, error) {
	p.mu.Lock()
	defer p.mu.Unlock()
	return cloneModels(p.models), nil
}

func (p *Provider) Chat(ctx context.Context, req *llm.Request) (*llm.Response, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	st := p.next(req)
	switch st.kind {
	case stepResponse:
		return cloneResponse(st.resp), nil
	case stepError:
		return nil, st.err
	case stepStream:
		return nil, fmt.Errorf("%w: llmtest stream step consumed by Chat", llm.ErrBadRequest)
	default:
		return nil, fmt.Errorf("%w: llmtest exhausted script", llm.ErrBadRequest)
	}
}

func (p *Provider) ChatStream(ctx context.Context, req *llm.Request) iter.Seq2[llm.Event, error] {
	st := p.next(req)
	var consumedMu sync.Mutex
	consumed := false
	return func(yield func(llm.Event, error) bool) {
		consumedMu.Lock()
		if consumed {
			consumedMu.Unlock()
			yield(nil, fmt.Errorf("%w: llmtest stream already consumed", llm.ErrBadRequest))
			return
		}
		consumed = true
		consumedMu.Unlock()

		if err := ctx.Err(); err != nil {
			yield(nil, err)
			return
		}
		switch st.kind {
		case stepStream:
			for _, event := range st.events {
				if err := ctx.Err(); err != nil {
					yield(nil, err)
					return
				}
				if !yield(cloneEvent(event), nil) {
					return
				}
			}
		case stepError:
			yield(nil, st.err)
		case stepResponse:
			yield(nil, fmt.Errorf("%w: llmtest response step consumed by ChatStream", llm.ErrBadRequest))
		default:
			yield(nil, fmt.Errorf("%w: llmtest exhausted script", llm.ErrBadRequest))
		}
	}
}

// EnqueueResponse appends a canned Chat response.
func (p *Provider) EnqueueResponse(r *llm.Response) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.steps = append(p.steps, step{kind: stepResponse, resp: cloneResponse(r)})
}

// EnqueueStream appends a canned ChatStream event sequence.
func (p *Provider) EnqueueStream(events ...llm.Event) {
	p.mu.Lock()
	defer p.mu.Unlock()
	copied := make([]llm.Event, len(events))
	for i, event := range events {
		copied[i] = cloneEvent(event)
	}
	p.steps = append(p.steps, step{kind: stepStream, events: copied})
}

// EnqueueError appends an error step for either Chat or ChatStream.
func (p *Provider) EnqueueError(err error) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.steps = append(p.steps, step{kind: stepError, err: err})
}

// Requests returns defensive copies of recorded requests.
func (p *Provider) Requests() []*llm.Request {
	p.mu.Lock()
	defer p.mu.Unlock()
	out := make([]*llm.Request, len(p.requests))
	for i, req := range p.requests {
		out[i] = cloneRequest(req)
	}
	return out
}

func (p *Provider) next(req *llm.Request) step {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.requests = append(p.requests, cloneRequest(req))
	if len(p.steps) == 0 {
		return step{}
	}
	st := p.steps[0]
	copy(p.steps, p.steps[1:])
	p.steps[len(p.steps)-1] = step{}
	p.steps = p.steps[:len(p.steps)-1]
	return st
}

func cloneRequest(req *llm.Request) *llm.Request {
	if req == nil {
		return nil
	}
	copied := *req
	copied.Messages = cloneMessages(req.Messages)
	copied.SystemCache = cloneCache(req.SystemCache)
	copied.StopSequences = append([]string(nil), req.StopSequences...)
	copied.Tools = cloneTools(req.Tools)
	if req.ResponseFormat != nil {
		format := *req.ResponseFormat
		format.Schema = cloneJSONLike(format.Schema)
		copied.ResponseFormat = &format
	}
	return &copied
}

func cloneResponse(resp *llm.Response) *llm.Response {
	if resp == nil {
		return nil
	}
	copied := *resp
	copied.Parts = cloneParts(resp.Parts)
	copied.Usage = cloneUsage(resp.Usage)
	copied.DroppedToolCalls = append([]llm.DroppedToolCall(nil), resp.DroppedToolCalls...)
	return &copied
}

func cloneMessages(msgs []llm.Message) []llm.Message {
	if len(msgs) == 0 {
		return nil
	}
	out := make([]llm.Message, len(msgs))
	for i, msg := range msgs {
		out[i] = msg
		out[i].Parts = cloneParts(msg.Parts)
	}
	return out
}

func cloneParts(parts []llm.Part) []llm.Part {
	if len(parts) == 0 {
		return nil
	}
	out := make([]llm.Part, len(parts))
	for i, part := range parts {
		out[i] = clonePart(part)
	}
	return out
}

func clonePart(part llm.Part) llm.Part {
	switch p := part.(type) {
	case llm.TextPart:
		p.Cache = cloneCache(p.Cache)
		return p
	case llm.ImagePart:
		p.Data = append([]byte(nil), p.Data...)
		p.Cache = cloneCache(p.Cache)
		return p
	case llm.FilePart:
		p.Data = append([]byte(nil), p.Data...)
		p.Cache = cloneCache(p.Cache)
		return p
	case llm.ToolCallPart:
		p.Args = append(json.RawMessage(nil), p.Args...)
		return p
	case llm.ToolResultPart:
		p.Content = cloneParts(p.Content)
		return p
	case llm.ReasoningPart:
		p.Raw = append(json.RawMessage(nil), p.Raw...)
		return p
	case llm.UnknownPart:
		p.Data = append(json.RawMessage(nil), p.Data...)
		return p
	case *llm.TextPart:
		if p == nil {
			return p
		}
		copied := *p
		copied.Cache = cloneCache(p.Cache)
		return &copied
	case *llm.ImagePart:
		if p == nil {
			return p
		}
		copied := *p
		copied.Data = append([]byte(nil), p.Data...)
		copied.Cache = cloneCache(p.Cache)
		return &copied
	case *llm.FilePart:
		if p == nil {
			return p
		}
		copied := *p
		copied.Data = append([]byte(nil), p.Data...)
		copied.Cache = cloneCache(p.Cache)
		return &copied
	case *llm.ToolCallPart:
		if p == nil {
			return p
		}
		copied := *p
		copied.Args = append(json.RawMessage(nil), p.Args...)
		return &copied
	case *llm.ToolResultPart:
		if p == nil {
			return p
		}
		copied := *p
		copied.Content = cloneParts(p.Content)
		return &copied
	case *llm.ReasoningPart:
		if p == nil {
			return p
		}
		copied := *p
		copied.Raw = append(json.RawMessage(nil), p.Raw...)
		return &copied
	case *llm.UnknownPart:
		if p == nil {
			return p
		}
		copied := *p
		copied.Data = append(json.RawMessage(nil), p.Data...)
		return &copied
	default:
		return part
	}
}

func cloneEvent(event llm.Event) llm.Event {
	switch e := event.(type) {
	case llm.ToolCallDelta:
		return e
	case *llm.ToolCallDelta:
		if e == nil {
			return e
		}
		copied := *e
		return &copied
	case llm.MessageStart:
		return e
	case *llm.MessageStart:
		if e == nil {
			return e
		}
		copied := *e
		return &copied
	case llm.TextDelta:
		return e
	case *llm.TextDelta:
		if e == nil {
			return e
		}
		copied := *e
		return &copied
	case llm.ReasoningDelta:
		e.Raw = append(json.RawMessage(nil), e.Raw...)
		return e
	case *llm.ReasoningDelta:
		if e == nil {
			return e
		}
		copied := *e
		copied.Raw = append(json.RawMessage(nil), e.Raw...)
		return &copied
	case llm.ToolCallStart:
		return e
	case *llm.ToolCallStart:
		if e == nil {
			return e
		}
		copied := *e
		return &copied
	case llm.ToolCallEnd:
		return e
	case *llm.ToolCallEnd:
		if e == nil {
			return e
		}
		copied := *e
		return &copied
	case llm.ToolCallDropped:
		return e
	case *llm.ToolCallDropped:
		if e == nil {
			return e
		}
		copied := *e
		return &copied
	case llm.MessageEnd:
		e.Usage = cloneUsage(e.Usage)
		return e
	case *llm.MessageEnd:
		if e == nil {
			return e
		}
		copied := *e
		copied.Usage = cloneUsage(e.Usage)
		return &copied
	default:
		return event
	}
}

func cloneTools(tools []llm.Tool) []llm.Tool {
	if len(tools) == 0 {
		return nil
	}
	out := make([]llm.Tool, len(tools))
	for i, tool := range tools {
		out[i] = tool
		out[i].InputSchema = cloneJSONLike(tool.InputSchema)
	}
	return out
}

func cloneCache(cache *llm.CacheHint) *llm.CacheHint {
	if cache == nil {
		return nil
	}
	copied := *cache
	return &copied
}

func cloneUsage(usage llm.Usage) llm.Usage {
	if usage.CostUSD != nil {
		cost := *usage.CostUSD
		usage.CostUSD = &cost
	}
	return usage
}

func cloneModels(models []llm.ModelInfo) []llm.ModelInfo {
	if len(models) == 0 {
		return nil
	}
	out := make([]llm.ModelInfo, len(models))
	for i, model := range models {
		out[i] = model
		if model.Pricing != nil {
			pricing := *model.Pricing
			out[i].Pricing = &pricing
		}
	}
	return out
}

func cloneJSONLike(value any) any {
	switch v := value.(type) {
	case json.RawMessage:
		return append(json.RawMessage(nil), v...)
	case []byte:
		return append([]byte(nil), v...)
	default:
		return value
	}
}
