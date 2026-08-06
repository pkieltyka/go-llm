GO ?= go
PNPM ?= pnpm

.DEFAULT_GOAL := build

.PHONY: build test e2e-test models

build:
	$(GO) build ./...
	mkdir -p bin
	$(GO) build -o bin/llm-cli ./cmd/llm-cli

test:
	$(GO) test -race -count=1 ./... $(TEST_ARGS)

e2e-test:
	$(GO) test -count=1 -tags=live -v ./internal/e2e $(E2E_ARGS)

models:
	$(PNPM) --dir scripts install --frozen-lockfile
	$(PNPM) --dir scripts run snapshot-models-table
