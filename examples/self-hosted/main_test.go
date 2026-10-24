package main

import "testing"

// TestOfflineRun runs the example against its in-process server with both
// the generic Chat Completions engine and the vLLM preset.
func TestOfflineRun(t *testing.T) {
	for _, server := range []string{"", "vllm"} {
		t.Setenv("LLM_BASE_URL", "")
		t.Setenv("LLM_API_KEY", "")
		t.Setenv("LLM_SERVER", server)
		main()
	}
}
