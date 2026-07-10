package llm

import (
	"net/http"
	"testing"
	"time"
)

func TestDefaultHTTPClient(t *testing.T) {
	client := DefaultHTTPClient()
	if client == nil {
		t.Fatalf("DefaultHTTPClient returned nil")
	}
	other := DefaultHTTPClient()
	if client == other {
		t.Fatalf("DefaultHTTPClient returned shared mutable client")
	}
	if client.Transport != other.Transport {
		t.Fatalf("default clients do not share their private transport")
	}
	if client.Timeout != 0 {
		t.Fatalf("Timeout = %s, want 0 for streaming safety", client.Timeout)
	}
	transport, ok := client.Transport.(*http.Transport)
	if !ok {
		t.Fatalf("Transport = %T, want *http.Transport", client.Transport)
	}
	if transport.ResponseHeaderTimeout != 120*time.Second {
		t.Fatalf("ResponseHeaderTimeout = %s", transport.ResponseHeaderTimeout)
	}
	if transport.IdleConnTimeout != 30*time.Second {
		t.Fatalf("IdleConnTimeout = %s", transport.IdleConnTimeout)
	}
	if transport.MaxIdleConnsPerHost != 16 {
		t.Fatalf("MaxIdleConnsPerHost = %d, want 16", transport.MaxIdleConnsPerHost)
	}
	client.Timeout = time.Second
	if other.Timeout != 0 {
		t.Fatalf("mutating one client changed another timeout to %s", other.Timeout)
	}
}

func TestDefaultHTTPTransportDarwinDisablesKeepAlives(t *testing.T) {
	if !defaultHTTPTransportForGOOS("darwin").DisableKeepAlives {
		t.Fatalf("darwin transport should disable keep-alives")
	}
	if defaultHTTPTransportForGOOS("linux").DisableKeepAlives {
		t.Fatalf("non-darwin transport should keep default keep-alive behavior")
	}
}
