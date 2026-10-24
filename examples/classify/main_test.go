package main

import "testing"

// TestOfflineRun runs the example's offline path end to end.
func TestOfflineRun(t *testing.T) {
	t.Setenv("OPENROUTER_API_KEY", "")
	main()
}
