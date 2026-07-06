package llm

import (
	"net/http"
	"runtime"
	"sync"
	"time"
)

var defaultHTTPClient = struct {
	once   sync.Once
	client *http.Client
}{}

// DefaultHTTPClient returns the shared HTTP client used by provider
// constructors when no custom client is supplied. Treat the returned client as
// immutable; it is tuned for long-lived LLM responses and streaming.
func DefaultHTTPClient() *http.Client {
	defaultHTTPClient.once.Do(func() {
		defaultHTTPClient.client = &http.Client{
			Transport: defaultHTTPTransportForGOOS(runtime.GOOS),
			Timeout:   0,
		}
	})
	return defaultHTTPClient.client
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
