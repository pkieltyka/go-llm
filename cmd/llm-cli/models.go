package main

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"strings"
	"text/tabwriter"

	llm "github.com/pkieltyka/go-llm"
)

func (a app) runModels(ctx context.Context, cfg modelsConfig) error {
	provider, err := a.providerFactory(ctx, providerConfigFromModels(cfg, a.stderr))
	if err != nil {
		return err
	}
	models, err := provider.Models(ctx)
	if err != nil {
		return err
	}
	rows := modelRows(models)
	if cfg.jsonOutput {
		data, err := json.MarshalIndent(rows, "", "  ")
		if err != nil {
			return err
		}
		if _, err := fmt.Fprintln(a.stdout, string(data)); err != nil {
			return fmt.Errorf("write models: %w", err)
		}
		return nil
	}
	tw := tabwriter.NewWriter(a.stdout, 0, 0, 2, ' ', 0)
	fmt.Fprintln(tw, "ID\tDISPLAY\tCONTEXT\tMAX OUTPUT\tINPUT $/M\tOUTPUT $/M\tCACHE READ $/M\tCACHE WRITE $/M\tEFFORTS\tDEFAULT EFFORT\tREASONING REQUIRED\tCAPABILITIES")
	for _, row := range rows {
		fmt.Fprintf(tw, "%s\t%s\t%d\t%d\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n",
			row.ID,
			row.DisplayName,
			row.ContextWindow,
			row.MaxOutputTokens,
			row.InputPerMTok,
			row.OutputPerMTok,
			row.CacheReadPerMTok,
			row.CacheWritePerMTok,
			joinOptionalModelMetadata(row.SupportedEfforts),
			row.DefaultEffort,
			formatReasoningRequired(row.ReasoningRequired),
			joinModelMetadata(row.Capabilities),
		)
	}
	return tw.Flush()
}

type modelRow struct {
	ID                string           `json:"id"`
	CanonicalID       string           `json:"canonical_id,omitempty"`
	DisplayName       string           `json:"display_name,omitempty"`
	ContextWindow     int              `json:"context_window,omitempty"`
	MaxOutputTokens   int              `json:"max_output_tokens,omitempty"`
	InputPerMTok      string           `json:"input_per_mtok,omitempty"`
	OutputPerMTok     string           `json:"output_per_mtok,omitempty"`
	CacheReadPerMTok  string           `json:"cache_read_per_mtok,omitempty"`
	CacheWritePerMTok string           `json:"cache_write_per_mtok,omitempty"`
	SupportedEfforts  *[]llm.Effort    `json:"supported_efforts,omitempty"`
	DefaultEffort     llm.Effort       `json:"default_effort,omitempty"`
	ReasoningRequired bool             `json:"reasoning_required,omitempty"`
	Capabilities      []llm.Capability `json:"capabilities,omitempty"`
}

func modelRows(models []llm.ModelInfo) []modelRow {
	rows := make([]modelRow, len(models))
	for i, model := range models {
		rows[i] = modelRow{
			ID:                model.ID,
			CanonicalID:       model.CanonicalID,
			DisplayName:       model.DisplayName,
			ContextWindow:     model.ContextWindow,
			MaxOutputTokens:   model.MaxOutputTokens,
			DefaultEffort:     model.DefaultEffort,
			ReasoningRequired: model.ReasoningRequired,
			Capabilities:      append([]llm.Capability(nil), model.Capabilities...),
		}
		if model.SupportedEfforts != nil {
			efforts := append(make([]llm.Effort, 0, len(model.SupportedEfforts)), model.SupportedEfforts...)
			rows[i].SupportedEfforts = &efforts
		}
		if model.Pricing != nil {
			if model.Pricing.HasInputPrice() {
				rows[i].InputPerMTok = formatModelPrice(model.Pricing.InputPerMTok)
			}
			if model.Pricing.HasOutputPrice() {
				rows[i].OutputPerMTok = formatModelPrice(model.Pricing.OutputPerMTok)
			}
			if model.Pricing.HasCacheReadPrice() {
				rows[i].CacheReadPerMTok = formatModelPrice(model.Pricing.CacheReadPerMTok)
			}
			if model.Pricing.HasCacheWritePrice() {
				rows[i].CacheWritePerMTok = formatModelPrice(model.Pricing.CacheWritePerMTok)
			}
		}
	}
	return rows
}

func formatModelPrice(price float64) string {
	if price < 0 || math.IsNaN(price) || math.IsInf(price, 0) {
		return ""
	}
	return formatFloat(price)
}

func formatReasoningRequired(required bool) string {
	if required {
		return "true"
	}
	return ""
}

func joinModelMetadata[T ~string](values []T) string {
	formatted := make([]string, len(values))
	for i, value := range values {
		formatted[i] = string(value)
	}
	return strings.Join(formatted, ",")
}

func joinOptionalModelMetadata[T ~string](values *[]T) string {
	if values == nil {
		return ""
	}
	return joinModelMetadata(*values)
}
