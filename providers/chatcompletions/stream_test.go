package chatcompletions

import (
	"context"
	"errors"
	"io"
	"net"
	"net/http"
	"strings"
	"testing"

	llm "github.com/pkieltyka/go-llm"
)

func TestConfiguredRetriesDoNotReplayAmbiguousFailures(t *testing.T) {
	for _, failure := range []struct {
		name      string
		roundTrip func(*http.Request) (*http.Response, error)
	}{
		{
			name: "post-send connection reset",
			roundTrip: func(*http.Request) (*http.Response, error) {
				return nil, &net.OpError{Op: "read", Net: "tcp", Err: errors.New("connection reset by peer")}
			},
		},
		{
			name: "ambiguous 500",
			roundTrip: func(req *http.Request) (*http.Response, error) {
				return &http.Response{
					StatusCode: http.StatusInternalServerError,
					Header:     http.Header{"Content-Type": []string{"application/json"}},
					Body:       io.NopCloser(strings.NewReader(`{"error":{"message":"failed after send","type":"server_error"}}`)),
					Request:    req,
				}, nil
			},
		},
	} {
		for _, path := range []string{"chat", "stream"} {
			t.Run(failure.name+"/"+path, func(t *testing.T) {
				calls := 0
				client := &http.Client{Transport: streamTestRoundTripFunc(func(req *http.Request) (*http.Response, error) {
					calls++
					return failure.roundTrip(req)
				})}
				provider, err := New("https://provider.test/v1", WithAPIKey("test"), WithHTTPClient(client), WithMaxRetries(2))
				if err != nil {
					t.Fatalf("New returned error: %v", err)
				}
				req := &llm.Request{Model: "test-model", Messages: []llm.Message{llm.UserText("ping")}}
				if path == "chat" {
					_, err = provider.Chat(context.Background(), req)
				} else {
					_, err = llm.Collect(provider.ChatStream(context.Background(), req))
				}
				if !errors.Is(err, llm.ErrServer) {
					t.Fatalf("error = %v, want ErrServer", err)
				}
				if calls != 1 {
					t.Fatalf("calls = %d, want exactly one non-replayed request", calls)
				}
			})
		}
	}
}

type streamTestRoundTripFunc func(*http.Request) (*http.Response, error)

func (fn streamTestRoundTripFunc) RoundTrip(req *http.Request) (*http.Response, error) {
	return fn(req)
}

func TestWithStreamEnabledPreservesGiantIntegers(t *testing.T) {
	// Splicing "stream": true must not route numbers through float64 —
	// integers beyond 2^53 have to survive verbatim.
	body := []byte(`{"model":"m","seed":9007199254740993,"max_tokens":123}`)
	out, err := withStreamEnabled(body)
	if err != nil {
		t.Fatalf("withStreamEnabled returned error: %v", err)
	}
	if !strings.Contains(string(out), `9007199254740993`) {
		t.Fatalf("giant integer mangled: %s", out)
	}
	if !strings.Contains(string(out), `"stream":true`) {
		t.Fatalf("stream flag missing: %s", out)
	}
}
