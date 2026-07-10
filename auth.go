package llm

import (
	"encoding/json"
	"fmt"
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
