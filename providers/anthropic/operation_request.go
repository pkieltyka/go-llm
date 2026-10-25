package anthropic

import (
	"fmt"
	sdk "github.com/anthropics/anthropic-sdk-go"
	sdkoption "github.com/anthropics/anthropic-sdk-go/option"
	llm "github.com/pkieltyka/go-llm"
	"github.com/pkieltyka/go-llm/providers/internal/providerutil"
	"strings"
)

const oauthSystemPreamble = "You are Claude Code, Anthropic's official CLI for Claude."

var claudeCodeTools = map[string]string{
	"read": "Read", "write": "Write", "edit": "Edit", "bash": "Bash", "grep": "Grep", "glob": "Glob",
	"askuserquestion": "AskUserQuestion", "enterplanmode": "EnterPlanMode", "exitplanmode": "ExitPlanMode",
	"killshell": "KillShell", "notebookedit": "NotebookEdit", "skill": "Skill", "task": "Task",
	"taskoutput": "TaskOutput", "todowrite": "TodoWrite", "webfetch": "WebFetch", "websearch": "WebSearch",
}

func oauthToolName(name string) string {
	if canonical, ok := claudeCodeTools[strings.ToLower(name)]; ok {
		return canonical
	}
	return name
}

func (p *Provider) buildOperationParams(req *llm.Request, op *preparedOperation, stream bool) (sdk.MessageNewParams, []sdkoption.RequestOption, error) {
	if err := validateRequest(req, stream); err != nil {
		return sdk.MessageNewParams{}, nil, err
	}
	if op.oauth {
		var err error
		req, op.reverseTools, err = oauthRequest(req)
		if err != nil {
			return sdk.MessageNewParams{}, nil, err
		}
	}
	params, opts, err := p.buildParamsAfterValidation(req)
	if err != nil {
		return params, opts, err
	}
	if op.oauth {
		preamble := sdk.TextBlockParam{Text: oauthSystemPreamble}
		effective := providerutil.ApplyCacheRetention(req)
		if effective.SystemCache != nil {
			preamble.CacheControl = cacheControl(effective.SystemCache)
		}
		params.System = append([]sdk.TextBlockParam{preamble}, params.System...)
	}
	return params, opts, nil
}

// Clone only fields we rewrite. Schema/argument/cache data is never mutated.
func oauthRequest(req *llm.Request) (*llm.Request, map[string]string, error) {
	out := *req
	reverse := map[string]string{}
	register := func(name string) error {
		folded := strings.ToLower(name)
		if old, ok := reverse[folded]; ok && old != name {
			return fmt.Errorf("%w: ambiguous case-insensitive Anthropic OAuth tool names", llm.ErrBadRequest)
		}
		reverse[folded] = name
		return nil
	}
	tools := func(input []llm.Tool) ([]llm.Tool, error) {
		result := append([]llm.Tool(nil), input...)
		for i := range result {
			if err := register(result[i].Name); err != nil {
				return nil, err
			}
			result[i].Name = oauthToolName(result[i].Name)
		}
		return result, nil
	}
	var err error
	out.Tools, err = tools(req.Tools)
	if err != nil {
		return nil, nil, err
	}
	out.ToolChoice.Name = oauthToolName(req.ToolChoice.Name)
	out.Messages = append([]llm.Message(nil), req.Messages...)
	for i, msg := range out.Messages {
		parts := append([]llm.Part(nil), msg.Parts...)
		for j, part := range parts {
			switch value := providerutil.DerefPart(part).(type) {
			case llm.ToolCallPart:
				value.Name = oauthToolName(value.Name)
				parts[j] = value
			case llm.ToolsDeltaPart:
				value.Added, err = tools(value.Added)
				if err != nil {
					return nil, nil, err
				}
				value.Removed = append([]string(nil), value.Removed...)
				for k := range value.Removed {
					value.Removed[k] = oauthToolName(value.Removed[k])
				}
				parts[j] = value
			}
		}
		out.Messages[i].Parts = parts
	}
	return &out, reverse, nil
}
