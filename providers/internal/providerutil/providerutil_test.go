package providerutil

import (
	"bytes"
	"context"
	"log/slog"
	"strings"
	"testing"
	"time"

	llm "github.com/pkieltyka/go-llm"
)

func TestLogFailureUsesSafeProviderErrorSummary(t *testing.T) {
	var output bytes.Buffer
	logger := slog.New(slog.NewTextHandler(&output, nil))
	err := &llm.ProviderError{
		Provider:   "openai",
		HTTPStatus: 400,
		Code:       "secret-code",
		Message:    "provider echoed a secret prompt",
		RawBody:    []byte("secret-body"),
		Kind:       llm.ErrBadRequest,
	}
	LogFailure(context.Background(), logger, "openai", &llm.Request{Model: "gpt-test"}, time.Now(), err)
	got := output.String()
	for _, want := range []string{"llm provider call failed", "provider=openai", "model=gpt-test", `error="llm/openai: 400 (llm: bad request)"`} {
		if !strings.Contains(got, want) {
			t.Fatalf("log output %q missing %q", got, want)
		}
	}
	for _, secret := range []string{"secret-code", "secret prompt", "secret-body"} {
		if strings.Contains(got, secret) {
			t.Fatalf("log output %q contains %q", got, secret)
		}
	}
}
