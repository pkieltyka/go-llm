// Package anthropic implements the llm.Provider interface for Anthropic's
// Messages API using API-key authentication.
//
// Mid-conversation RoleSystem messages are sent as role "system" messages,
// and ToolsDeltaPart changes as tool_addition/tool_removal blocks under the
// inline-tools beta, for models whose llm.ModelInfo.Behavior allows it;
// other models get them folded into the system prompt and tool list.
package anthropic
