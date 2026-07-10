package llm

import (
	"net/http"
	"runtime"
	"sync"
	"time"
)

var defaultHTTPTransport = struct {
	once      sync.Once
	transport *http.Transport
}{}

// DefaultHTTPClient returns a fresh HTTP client tuned for long-lived LLM
// responses and streaming. Clients share a private connection-pooling
// transport, so changing client-level fields does not affect other callers.
func DefaultHTTPClient() *http.Client {
	defaultHTTPTransport.once.Do(func() {
		defaultHTTPTransport.transport = defaultHTTPTransportForGOOS(runtime.GOOS)
	})
	return &http.Client{Transport: defaultHTTPTransport.transport}
}

func defaultHTTPTransportForGOOS(goos string) *http.Transport {
	base, ok := http.DefaultTransport.(*http.Transport)
	if !ok {
		base = &http.Transport{}
	}
	transport := base.Clone()
	transport.ResponseHeaderTimeout = 120 * time.Second
	transport.IdleConnTimeout = 30 * time.Second
	transport.MaxIdleConnsPerHost = 16
	if goos == "darwin" {
		transport.DisableKeepAlives = true
	}
	return transport
}
