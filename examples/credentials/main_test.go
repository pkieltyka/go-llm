package main

import "testing"

// TestOfflineRun runs the example against its in-process server.
func TestOfflineRun(t *testing.T) {
	t.Setenv("OPENROUTER_API_KEY", "")
	t.Setenv("GO_LLM_DEMO_KEY", "")
	main()
}
