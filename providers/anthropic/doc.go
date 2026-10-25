// Package anthropic implements the llm.Provider interface for Anthropic's
// Messages API using API keys or typed Pi-compatible OAuth credentials.
// OAuth refresh is serialized and persisted through WithCredentials. Initial
// login returns a credential to the host via NewLoginFlow. Remote browsers need
// forwarding to the local launch URL, including copy-code login.
//
// OAuth requests use Pi's pinned headers and request-only system/tool
// adaptations. Raw Client SDK requests fail closed for OAuth: use the prepared
// Chat, ChatStream, Models and CountTokens operations. API-key SDK access is
// unchanged. OAuth support does not establish account eligibility or billing.
//
// Mid-conversation RoleSystem messages are sent as role "system" messages,
// and ToolsDeltaPart changes as tool_addition/tool_removal blocks under the
// inline-tools beta, for models whose llm.ModelInfo.Behavior allows it;
// other models get them folded into the system prompt and tool list.
package anthropic
