package openai_test

import (
	"encoding/json"
	"reflect"
	"strings"
	"testing"

	llm "github.com/pkieltyka/go-llm"
	"github.com/pkieltyka/go-llm/providers/openai"
)

var publicStore = false

// This literal is a compile-time consumer check: ordinary request options use
// only go-llm, openai package, and standard-library types.
var publicOptions = openai.Options{
	Store:                &publicStore,
	Conversation:         &openai.Conversation{ID: "conv_1"},
	Include:              []openai.Include{openai.IncludeMessageOutputTextLogprobs},
	HostedTools:          []json.RawMessage{json.RawMessage(`{"type":"web_search"}`)},
	Verbosity:            openai.VerbosityLow,
	Metadata:             openai.Metadata{"purpose": "test"},
	ServiceTier:          openai.ServiceTierDefault,
	PromptCacheRetention: openai.PromptCacheRetention24h,
}

var _ llm.ProviderOptions = publicOptions

func ExampleOptions() {
	store := false
	req := &llm.Request{
		Model:    "gpt-5.5",
		Messages: []llm.Message{llm.UserText("Find the current release notes.")},
		ProviderOptions: openai.Options{
			Store:       &store,
			HostedTools: []json.RawMessage{json.RawMessage(`{"type":"web_search"}`)},
			Verbosity:   openai.VerbosityLow,
		},
	}
	_ = req
}

func TestOptionsPublicFieldsDoNotExposeVendorSDKTypes(t *testing.T) {
	assertNoOpenAISDKType(t, reflect.TypeOf(openai.Options{}), map[reflect.Type]bool{})
}

func assertNoOpenAISDKType(t *testing.T, typ reflect.Type, seen map[reflect.Type]bool) {
	t.Helper()
	if typ == nil || seen[typ] {
		return
	}
	seen[typ] = true
	if strings.HasPrefix(typ.PkgPath(), "github.com/openai/openai-go") {
		t.Fatalf("ordinary Options exposes vendor SDK type %s", typ)
	}
	switch typ.Kind() {
	case reflect.Array, reflect.Pointer, reflect.Slice:
		assertNoOpenAISDKType(t, typ.Elem(), seen)
	case reflect.Map:
		assertNoOpenAISDKType(t, typ.Key(), seen)
		assertNoOpenAISDKType(t, typ.Elem(), seen)
	case reflect.Struct:
		for i := 0; i < typ.NumField(); i++ {
			assertNoOpenAISDKType(t, typ.Field(i).Type, seen)
		}
	}
}
