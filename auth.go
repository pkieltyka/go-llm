package llm

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"net/url"
	"os"
)

// AuthFile maps provider IDs to explicit credentials loaded by the caller.
type AuthFile map[string]AuthCredential

// AuthCredential is the pi-compatible credential union accepted by LoadAuthFile.
type AuthCredential struct {
	Type      string
	Key       string
	Access    string
	Refresh   string
	Expires   int64
	AccountID string
	Model     string
	BaseURL   string
}

// String redacts credential material from formatted output.
func (AuthCredential) String() string { return "llm: auth credential (redacted)" }

// GoString redacts credential material from Go-syntax formatted output.
func (AuthCredential) GoString() string { return "llm.AuthCredential{redacted}" }

// LogValue redacts credential material from structured logs.
func (AuthCredential) LogValue() slog.Value {
	return slog.StringValue("llm: auth credential (redacted)")
}

// OAuthPersistenceFunc durably persists a renewed OAuth credential before it
// becomes visible to provider requests. Implementations MUST honor ctx and
// return only after persistence is durable; returning an error prevents the
// provider from publishing the renewed credential. Refreshable credentials
// require a non-nil callback. A context-aware no-op explicitly opts into
// in-memory-only rotation, which can leave stored refresh tokens stale after a
// restart.
type OAuthPersistenceFunc func(context.Context, AuthCredential) error

// LoginFlow is a single-use interactive provider login. Begin prepares the
// provider authorization and returns a non-secret launch target. Complete
// waits for provider completion delivered automatically or through Submit.
// Submit is an optional provider-specific manual or fallback path. Cancel is
// idempotent and interrupts pending waits and network work.
//
// Implementations own all provider protocol details and sensitive ephemeral
// state. They must be safe for Complete, Submit, and Cancel to run from
// different goroutines, must honor context cancellation, and must redact
// credentials, authorization codes, PKCE verifiers, and CSRF state from
// errors and formatted output.
type LoginFlow interface {
	// Begin prepares the provider authorization and returns display data.
	Begin(ctx context.Context) (LoginAuthorization, error)
	// Complete waits for provider completion and returns the minted credential.
	Complete(ctx context.Context) (AuthCredential, error)
	// Submit forwards one complete provider-specific manual response to
	// Complete. Implementations may use it only as a fallback.
	Submit(ctx context.Context, response string) error
	// Cancel interrupts the flow and releases its resources.
	Cancel()
}

// LoginAuthorization describes how the host should begin a login without
// exposing the provider authorization URL or its ephemeral state. URL is a
// short-lived loopback launch target that redirects to the provider. It must
// be treated as transient UI data and must not be persisted.
type LoginAuthorization struct {
	url          string
	instructions string
}

// NewLoginAuthorization constructs transient login instructions for a
// provider implementation. url must be an absolute HTTP(S) URL.
func NewLoginAuthorization(launchURL, instructions string) (LoginAuthorization, error) {
	parsed, err := url.Parse(launchURL)
	if err != nil || (parsed.Scheme != "http" && parsed.Scheme != "https") || parsed.Host == "" {
		return LoginAuthorization{}, fmt.Errorf("%w: login authorization URL must be absolute HTTP(S)", ErrBadRequest)
	}
	return LoginAuthorization{url: launchURL, instructions: instructions}, nil
}

// URL returns the transient, non-secret launch URL for the user's browser.
func (authorization LoginAuthorization) URL() string { return authorization.url }

// Instructions returns provider-neutral display guidance for the host.
func (authorization LoginAuthorization) Instructions() string { return authorization.instructions }

// String redacts the transient launch target from formatted output.
func (LoginAuthorization) String() string { return "llm: login authorization (redacted)" }

// GoString redacts the transient launch target from Go-syntax formatting.
func (LoginAuthorization) GoString() string { return "llm.LoginAuthorization{redacted}" }

// LogValue redacts the transient launch target from structured logs.
func (LoginAuthorization) LogValue() slog.Value {
	return slog.StringValue("llm: login authorization (redacted)")
}

// LoadAuthFile parses a pi-compatible credential file from path.
func LoadAuthFile(path string) (AuthFile, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	return ParseAuthFile(data)
}

// ParseAuthFile parses either a bare provider credential map or a
// {"providers": ...} wrapper.
func ParseAuthFile(data []byte) (AuthFile, error) {
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(data, &raw); err != nil {
		return nil, err
	}
	if raw == nil {
		return AuthFile{}, nil
	}

	if providersRaw, ok := raw["providers"]; ok {
		var providers AuthFile
		if err := json.Unmarshal(providersRaw, &providers); err != nil {
			return nil, fmt.Errorf("parse auth providers: %w", err)
		}
		if providers == nil {
			return AuthFile{}, nil
		}
		return providers, nil
	}

	var providers AuthFile
	if err := json.Unmarshal(data, &providers); err != nil {
		return nil, err
	}
	if providers == nil {
		return AuthFile{}, nil
	}
	return providers, nil
}

// UnmarshalJSON accepts pi's camelCase fields plus go-llm's snake_case e2e
// aliases. Unknown fields are intentionally ignored for forward compatibility.
func (c *AuthCredential) UnmarshalJSON(data []byte) error {
	type credential struct {
		Type      string      `json:"type"`
		Key       string      `json:"key"`
		APIKey    string      `json:"api_key"`
		Access    string      `json:"access"`
		Refresh   string      `json:"refresh"`
		Expires   json.Number `json:"expires"`
		AccountID string      `json:"accountId"`
		Model     string      `json:"model"`
		BaseURL   string      `json:"base_url"`
		BaseURL2  string      `json:"baseUrl"`
	}
	var in credential
	if err := json.Unmarshal(data, &in); err != nil {
		return err
	}
	next := AuthCredential{
		Type:      in.Type,
		Key:       in.Key,
		Access:    in.Access,
		Refresh:   in.Refresh,
		AccountID: in.AccountID,
		Model:     in.Model,
		BaseURL:   in.BaseURL,
	}
	if next.Key == "" {
		next.Key = in.APIKey
	}
	if next.BaseURL == "" {
		next.BaseURL = in.BaseURL2
	}
	if in.Expires != "" {
		expires, err := in.Expires.Int64()
		if err != nil {
			return fmt.Errorf("expires must be an integer millisecond epoch: %w", err)
		}
		next.Expires = expires
	}
	*c = next
	return nil
}
