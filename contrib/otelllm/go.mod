module github.com/pkieltyka/go-llm/contrib/otelllm

go 1.26.0

require (
	github.com/pkieltyka/go-llm v0.3.0
	go.opentelemetry.io/otel v1.47.0
	go.opentelemetry.io/otel/metric v1.47.0
	go.opentelemetry.io/otel/sdk v1.47.0
	go.opentelemetry.io/otel/sdk/metric v1.47.0
	go.opentelemetry.io/otel/trace v1.47.0
)

require (
	github.com/cespare/xxhash/v2 v2.3.0 // indirect
	github.com/go-logr/logr v1.4.4 // indirect
	github.com/go-logr/stdr v1.2.2 // indirect
	github.com/google/uuid v1.6.0 // indirect
	go.opentelemetry.io/auto/sdk v1.2.1 // indirect
	go.opentelemetry.io/otel/log v1.47.0 // indirect
	golang.org/x/sys v0.49.0 // indirect
)

// The module tracks go-llm at HEAD through this replace. The require above
// is a placeholder BELOW the real minimum (the module uses APIs added after
// v0.3.0): bump it to the matching go-llm tag before tagging this module
// (docs/release.md), or consumers will resolve a version that cannot build.
replace github.com/pkieltyka/go-llm => ../..
