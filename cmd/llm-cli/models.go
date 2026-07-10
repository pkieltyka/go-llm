package main

import (
	"context"
	"encoding/json"
	"fmt"
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
	fmt.Fprintln(tw, "ID\tDISPLAY\tCONTEXT\tMAX OUTPUT\tINPUT $/M\tOUTPUT $/M")
	for _, row := range rows {
		fmt.Fprintf(tw, "%s\t%s\t%d\t%d\t%s\t%s\n",
			row.ID,
			row.DisplayName,
			row.ContextWindow,
			row.MaxOutputTokens,
			row.InputPerMTok,
			row.OutputPerMTok,
		)
	}
	return tw.Flush()
}

type modelRow struct {
	ID              string `json:"id"`
	CanonicalID     string `json:"canonical_id,omitempty"`
	DisplayName     string `json:"display_name,omitempty"`
	ContextWindow   int    `json:"context_window,omitempty"`
	MaxOutputTokens int    `json:"max_output_tokens,omitempty"`
	InputPerMTok    string `json:"input_per_mtok,omitempty"`
	OutputPerMTok   string `json:"output_per_mtok,omitempty"`
}

func modelRows(models []llm.ModelInfo) []modelRow {
	rows := make([]modelRow, len(models))
	for i, model := range models {
		rows[i] = modelRow{
			ID:              model.ID,
			CanonicalID:     model.CanonicalID,
			DisplayName:     model.DisplayName,
			ContextWindow:   model.ContextWindow,
			MaxOutputTokens: model.MaxOutputTokens,
		}
		if model.Pricing != nil {
			rows[i].InputPerMTok = formatFloat(model.Pricing.InputPerMTok)
			rows[i].OutputPerMTok = formatFloat(model.Pricing.OutputPerMTok)
		}
	}
	return rows
}
