package llm_test

import (
	"bytes"
	"context"
	"encoding/json"
	"reflect"
	"testing"

	llm "github.com/pkieltyka/go-llm"
	"github.com/pkieltyka/go-llm/llmtest"
)

func costUsage(cost float64, source string) llm.Usage {
	return llm.Usage{InputTokens: 2, OutputTokens: 1, TotalTokens: 3, CostUSD: &cost, CostSource: source}
}

func TestUsageAddCostCompleteness(t *testing.T) {
	native := costUsage(0.5, llm.CostSourceNative)
	estimated := costUsage(0.25, llm.CostSourceEstimated)
	unknown := llm.Usage{InputTokens: 4, TotalTokens: 4}
	partial := costUsage(0.25, llm.CostSourceNative)
	partial.CostIncomplete = true
	for _, tt := range []struct {
		name       string
		a, b       llm.Usage
		cost       *float64
		source     string
		incomplete bool
	}{
		{"priced then unpriced", native, unknown, llm.Ptr(0.5), llm.CostSourceNative, true},
		{"unpriced then priced", unknown, native, llm.Ptr(0.5), llm.CostSourceNative, true},
		{"known zero", costUsage(0, llm.CostSourceNative), llm.Usage{}, llm.Ptr(0.0), llm.CostSourceNative, false},
		{"zero subtotal", costUsage(0, llm.CostSourceNative), unknown, llm.Ptr(0.0), llm.CostSourceNative, true},
		{"no price", unknown, unknown, nil, "", true},
		{"no usage", llm.Usage{}, llm.Usage{}, nil, "", false},
		{"mixed provenance", native, estimated, llm.Ptr(0.75), llm.CostSourceEstimated, false},
		{"explicit partial", partial, native, llm.Ptr(0.75), llm.CostSourceNative, true},
		{"explicit missing without tokens", llm.Usage{CostIncomplete: true}, native, llm.Ptr(0.5), llm.CostSourceNative, true},
		{"explicit missing only", llm.Usage{CostIncomplete: true}, llm.Usage{}, nil, "", true},
		{"unspecified legacy provenance", costUsage(0.25, ""), native, llm.Ptr(0.75), llm.CostSourceEstimated, false},
		{"one-hour subset only", llm.Usage{CacheWrite1hTokens: 1}, native, llm.Ptr(0.5), llm.CostSourceNative, true},
		{"reasoning subset only", llm.Usage{ReasoningTokens: 1}, native, llm.Ptr(0.5), llm.CostSourceNative, true},
	} {
		t.Run(tt.name, func(t *testing.T) {
			got := tt.a.Add(tt.b)
			if !reflect.DeepEqual(got.CostUSD, tt.cost) || got.CostSource != tt.source || got.CostIncomplete != tt.incomplete {
				t.Fatalf("Add cost = %v, source=%q incomplete=%v; want %v, %q, %v", got.CostUSD, got.CostSource, got.CostIncomplete, tt.cost, tt.source, tt.incomplete)
			}
			if got.TotalTokens != tt.a.TotalTokens+tt.b.TotalTokens || got.InputTokens != tt.a.InputTokens+tt.b.InputTokens {
				t.Fatalf("token sum = %+v", got)
			}
		})
	}
}

func TestUsageAddGroupingSurvivesJSON(t *testing.T) {
	inputs := []llm.Usage{
		{}, costUsage(0, llm.CostSourceNative), costUsage(0.5, llm.CostSourceNative),
		costUsage(0.25, llm.CostSourceEstimated), costUsage(0.25, ""),
		{TotalTokens: 7}, {CostIncomplete: true},
		{CacheWriteTokens: 2, CacheWrite1hTokens: 1},
		{CostUSD: llm.Ptr(0.5), CostSource: llm.CostSourceNative, CostIncomplete: true},
	}
	for _, a := range inputs {
		// Legacy nil-cost token values normalize to incomplete when aggregated;
		// empty usage preserves their accounting, including nil versus zero.
		if !reflect.DeepEqual(a.Add(llm.Usage{}), (llm.Usage{}).Add(a)) {
			t.Fatalf("empty identity differs for %+v", a)
		}
		for _, b := range inputs {
			if !reflect.DeepEqual(a.Add(b), b.Add(a)) {
				t.Fatalf("order changes accounting for %+v and %+v", a, b)
			}
			for _, c := range inputs {
				ab := a.Add(b)
				data, err := json.Marshal(ab)
				if err != nil {
					t.Fatal(err)
				}
				var persisted llm.Usage
				if err := json.Unmarshal(data, &persisted); err != nil {
					t.Fatal(err)
				}
				if got, want := persisted.Add(c), a.Add(b.Add(c)); !reflect.DeepEqual(got, want) {
					t.Fatalf("grouping/JSON changes accounting: (%+v + %+v) + %+v = %+v; want %+v", a, b, c, got, want)
				}
			}
		}
	}
}

func TestUsageCostCompletenessLegacyJSONAndClone(t *testing.T) {
	legacy := []byte(`{"input_tokens":2,"output_tokens":1,"total_tokens":3,"cache_read_tokens":0,"cache_write_tokens":0,"reasoning_tokens":0,"cost_usd":0,"cost_source":"native"}`)
	var usage llm.Usage
	if err := json.Unmarshal(legacy, &usage); err != nil {
		t.Fatal(err)
	}
	data, err := json.Marshal(usage)
	if err != nil || !bytes.Equal(data, legacy) || usage.CostIncomplete || usage.CostUSD == nil || *usage.CostUSD != 0 {
		t.Fatalf("legacy zero changed: %s, %+v, %v", data, usage, err)
	}
	usage.CostIncomplete = true
	response := &llm.Response{Usage: usage}
	clone := response.Clone()
	if !clone.Usage.CostIncomplete {
		t.Fatal("clone dropped completeness")
	}
	*clone.Usage.CostUSD = 3
	if *response.Usage.CostUSD != 0 {
		t.Fatal("clone shares cost storage")
	}
	data, err = json.Marshal(response)
	if err != nil || !bytes.Contains(data, []byte(`"cost_incomplete":true`)) {
		t.Fatalf("response missing completeness: %s, %v", data, err)
	}
	var decoded llm.Response
	if err := json.Unmarshal(data, &decoded); err != nil || !decoded.Usage.CostIncomplete {
		t.Fatalf("response lost completeness: %+v, %v", decoded, err)
	}
}

func TestUsageAddOwnsCostStorage(t *testing.T) {
	for _, inputs := range [][2]llm.Usage{
		{costUsage(0.5, llm.CostSourceNative), {}},
		{{}, costUsage(0.5, llm.CostSourceNative)},
		{costUsage(0.5, llm.CostSourceNative), costUsage(0.25, llm.CostSourceEstimated)},
	} {
		before, _ := json.Marshal(inputs)
		got := inputs[0].Add(inputs[1])
		*got.CostUSD = 9
		after, _ := json.Marshal(inputs)
		if !bytes.Equal(before, after) {
			t.Fatal("Add result aliases input cost")
		}
	}
}

func TestSessionAndTrackerRetainIncompleteSubtotal(t *testing.T) {
	p := llmtest.New(llmtest.WithName("fake"))
	tracker := llm.NewUsageTracker()
	session := llm.NewSession(llm.Wrap(p, tracker.Middleware()), "model-a")
	known := costUsage(0.5, llm.CostSourceNative)
	unknown := llm.EstimateCost(llm.Usage{CacheWriteTokens: 2, CacheWrite1hTokens: 1}, llm.ModelPricing{CacheWritePerMTok: 2})
	if unknown.CostUSD != nil {
		t.Fatal("one-hour writes acquired an invented price")
	}
	for _, usage := range []llm.Usage{known, unknown, {CostIncomplete: true}} {
		p.EnqueueResponse(&llm.Response{Parts: []llm.Part{llm.Text("ok")}, Usage: usage})
		if _, err := session.Chat(context.Background(), llm.Text("hi")); err != nil {
			t.Fatal(err)
		}
	}
	for _, usage := range []llm.Usage{session.Usage(), tracker.Stats().Usage, tracker.Stats().ByProviderModel["fake/model-a"].Usage} {
		if usage.CostUSD == nil || *usage.CostUSD != 0.5 || usage.CostSource != llm.CostSourceNative || !usage.CostIncomplete || usage.CacheWrite1hTokens != 1 {
			t.Fatalf("lost subtotal metadata: %+v", usage)
		}
	}
}

func TestEstimateCostPreservesExplicitIncompleteness(t *testing.T) {
	usage := llm.Usage{InputTokens: 1, CostIncomplete: true}
	got := llm.EstimateCost(usage, llm.ModelPricing{InputPerMTok: 1})
	if got.CostUSD == nil || !got.CostIncomplete || got.CostSource != llm.CostSourceEstimated {
		t.Fatalf("estimate erased explicit incompleteness: %+v", got)
	}
}
