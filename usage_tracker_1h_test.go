package llm

import "testing"

func TestSumUsageAddsOneHourCacheWrites(t *testing.T) {
	var state usageCostAggregation
	total := sumUsage(Usage{CacheWriteTokens: 5, CacheWrite1hTokens: 2}, Usage{CacheWriteTokens: 7, CacheWrite1hTokens: 3}, &state)
	if total.CacheWriteTokens != 12 || total.CacheWrite1hTokens != 5 {
		t.Fatalf("aggregated cache writes = %d/%d, want 12/5", total.CacheWriteTokens, total.CacheWrite1hTokens)
	}
}
