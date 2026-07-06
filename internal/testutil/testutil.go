// Package testutil hosts the small test helpers shared by the module's test
// suites (root, providers, engines): canonical JSON assertions, event-stream
// builders, and HTTP transport fakes. Test-only — nothing here ships in the
// public API.
package testutil

import (
	"bytes"
	"encoding/json"
	"iter"
	"net/http"
	"reflect"
	"testing"

	llm "github.com/pkieltyka/go-llm"
)

// MustCompactJSON marshals value and returns its compact JSON encoding.
func MustCompactJSON(t testing.TB, value any) string {
	t.Helper()
	raw, err := json.Marshal(value)
	if err != nil {
		t.Fatalf("Marshal returned error: %v", err)
	}
	var buf bytes.Buffer
	if err := json.Compact(&buf, raw); err != nil {
		t.Fatalf("Compact returned error: %v\n%s", err, raw)
	}
	return buf.String()
}

// AssertJSONEqual compares two JSON documents semantically (key order and
// whitespace insensitive) and fails with a pretty-printed diff of both sides.
func AssertJSONEqual(t testing.TB, got, want string) {
	t.Helper()
	var gotAny, wantAny any
	if err := json.Unmarshal([]byte(got), &gotAny); err != nil {
		t.Fatalf("got is invalid JSON: %v\n%s", err, got)
	}
	if err := json.Unmarshal([]byte(want), &wantAny); err != nil {
		t.Fatalf("want is invalid JSON: %v\n%s", err, want)
	}
	if !reflect.DeepEqual(gotAny, wantAny) {
		gotPretty, _ := json.MarshalIndent(gotAny, "", "  ")
		wantPretty, _ := json.MarshalIndent(wantAny, "", "  ")
		t.Fatalf("JSON mismatch\ngot:\n%s\nwant:\n%s", gotPretty, wantPretty)
	}
}

// EventSeq turns a fixed event list into an error-free stream.
func EventSeq(events ...llm.Event) iter.Seq2[llm.Event, error] {
	return func(yield func(llm.Event, error) bool) {
		for _, event := range events {
			if !yield(event, nil) {
				return
			}
		}
	}
}

// RoundTripFunc adapts a function into an http.RoundTripper.
type RoundTripFunc func(*http.Request) (*http.Response, error)

// RoundTrip implements http.RoundTripper.
func (f RoundTripFunc) RoundTrip(req *http.Request) (*http.Response, error) {
	return f(req)
}

// MapRawEvents is the shared stream-collect loop of the provider mapping
// tests: it unmarshals each raw JSON fixture into the provider SDK's stream
// event union E, runs it through the adapter's mapEvent, and returns the
// accumulated normalized events.
func MapRawEvents[E any](t testing.TB, raws []string, mapEvent func(E) ([]llm.Event, error)) []llm.Event {
	t.Helper()
	var events []llm.Event
	for _, raw := range raws {
		var event E
		if err := json.Unmarshal([]byte(raw), &event); err != nil {
			t.Fatalf("unmarshal stream event %s: %v", raw, err)
		}
		mapped, err := mapEvent(event)
		if err != nil {
			t.Fatalf("mapEvent returned error: %v", err)
		}
		events = append(events, mapped...)
	}
	return events
}

// CollectRawEvents maps raw fixture events (MapRawEvents) and collects them
// into a complete response.
func CollectRawEvents[E any](t testing.TB, raws []string, mapEvent func(E) ([]llm.Event, error)) *llm.Response {
	t.Helper()
	resp, err := llm.Collect(EventSeq(MapRawEvents(t, raws, mapEvent)...))
	if err != nil {
		t.Fatalf("Collect returned error: %v", err)
	}
	return resp
}
