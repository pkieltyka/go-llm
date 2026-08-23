GO ?= go
PNPM ?= pnpm
GOLANGCI_LINT_VERSION ?= v2.12.2
GOVULNCHECK_VERSION ?= v1.5.0

.DEFAULT_GOAL := build

.PHONY: build test e2e-test models check check-build check-vet check-lint \
	check-test check-snapshots check-live check-coverage \
	check-vulnerabilities check-fuzz

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

# check is the complete credential-free CI gate and is reused by releases.
check: check-build check-vet check-lint check-test check-snapshots check-live \
	check-coverage check-vulnerabilities check-fuzz

check-build: build
	test -x bin/llm-cli
	./bin/llm-cli --help >/dev/null
	./bin/llm-cli --version >/dev/null

check-vet:
	$(GO) vet ./...

check-lint:
	$(GO) run github.com/golangci/golangci-lint/v2/cmd/golangci-lint@$(GOLANGCI_LINT_VERSION) run --default=none --enable=govet --enable=ineffassign --enable=unused ./...

check-test: test

check-snapshots:
	$(PNPM) --dir scripts install --frozen-lockfile
	$(PNPM) --dir scripts test

check-live:
	$(GO) test -count=1 -tags=live -run '^TestLiveRunnerManifests$$' ./internal/e2e
	$(GO) test -count=1 -tags=live -run '^$$' ./...

check-coverage:
	./scripts/check-coverage_test.sh
	./scripts/check-coverage.sh

check-vulnerabilities:
	$(GO) run golang.org/x/vuln/cmd/govulncheck@$(GOVULNCHECK_VERSION) ./...

# go test -fuzz accepts only one package and one matching target per call.
check-fuzz:
	@set -eu; \
	for pkg in $$($(GO) list ./...); do \
		for target in $$($(GO) test -list '^Fuzz' "$$pkg" | grep '^Fuzz' || true); do \
			$(GO) test -run='^$$' -fuzz="^$${target}$$" -fuzztime=5s "$$pkg"; \
		done; \
	done
