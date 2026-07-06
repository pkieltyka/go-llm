package main

import (
	"bytes"
	"context"
	"errors"
	"strings"
	"testing"
)

func TestParseChatFlagsHelp(t *testing.T) {
	for _, arg := range []string{"-h", "--help"} {
		var stderr bytes.Buffer
		_, err := parseChatFlags([]string{arg}, &stderr)
		if !errors.Is(err, errHelp) {
			t.Fatalf("%s: err = %v, want errHelp", arg, err)
		}
		usage := stderr.String()
		for _, want := range []string{
			"Usage: llm-cli [flags] [prompt]",
			"llm-cli models [flags]",
			"forces the non-streaming path",
			"-provider",
			"OAuth access token",
			"Examples:",
			"llm-cli models -p openrouter",
		} {
			if !strings.Contains(usage, want) {
				t.Fatalf("%s usage missing %q:\n%s", arg, want, usage)
			}
		}
	}
}

func TestParseChatFlagsBadFlag(t *testing.T) {
	var stderr bytes.Buffer
	_, err := parseChatFlags([]string{"-bogus"}, &stderr)
	if !errors.Is(err, errUsage) {
		t.Fatalf("err = %v, want errUsage", err)
	}
	got := stderr.String()
	if !strings.Contains(got, "flag provided but not defined: -bogus") {
		t.Fatalf("stderr missing flag error: %q", got)
	}
	if !strings.Contains(got, "Usage: llm-cli") {
		t.Fatalf("stderr missing usage after bad flag: %q", got)
	}
}

func TestParseModelsFlagsHelp(t *testing.T) {
	var stderr bytes.Buffer
	_, err := parseModelsFlags([]string{"-h"}, &stderr)
	if !errors.Is(err, errHelp) {
		t.Fatalf("err = %v, want errHelp", err)
	}
	usage := stderr.String()
	for _, want := range []string{"Usage: llm-cli models [flags]", "-provider", "llm-cli models -p openrouter"} {
		if !strings.Contains(usage, want) {
			t.Fatalf("models usage missing %q:\n%s", want, usage)
		}
	}
}

func TestParseModelsFlagsBadInput(t *testing.T) {
	var stderr bytes.Buffer
	_, err := parseModelsFlags([]string{"-bogus"}, &stderr)
	if !errors.Is(err, errUsage) {
		t.Fatalf("bad flag err = %v, want errUsage", err)
	}
	if got := stderr.String(); !strings.Contains(got, "flag provided but not defined") {
		t.Fatalf("stderr missing flag error: %q", got)
	}

	stderr.Reset()
	_, err = parseModelsFlags([]string{"unexpected"}, &stderr)
	if !errors.Is(err, errUsage) {
		t.Fatalf("unexpected arg err = %v, want errUsage", err)
	}
	if got := stderr.String(); !strings.Contains(got, `unexpected argument "unexpected"`) {
		t.Fatalf("stderr missing unexpected-argument error: %q", got)
	}
}

// run must surface help as success (exit 0) and bad flags as errUsage
// (exit 1, already reported) through the app entrypoint.
func TestRunHelpAndBadFlagExitPaths(t *testing.T) {
	var stdout, stderr bytes.Buffer
	a := testApp(nil, &stdout, &stderr)
	if err := a.run(context.Background(), []string{"-h"}); err != nil {
		t.Fatalf("run -h err = %v, want nil (exit 0)", err)
	}
	if !strings.Contains(stderr.String(), "Usage: llm-cli") {
		t.Fatalf("run -h wrote no usage to stderr: %q", stderr.String())
	}

	stderr.Reset()
	if err := a.run(context.Background(), []string{"-bogus"}); !errors.Is(err, errUsage) {
		t.Fatalf("run -bogus err = %v, want errUsage", err)
	}

	stderr.Reset()
	if err := a.run(context.Background(), []string{"models", "-h"}); err != nil {
		t.Fatalf("run models -h err = %v, want nil (exit 0)", err)
	}
	if !strings.Contains(stderr.String(), "Usage: llm-cli models") {
		t.Fatalf("run models -h wrote no usage to stderr: %q", stderr.String())
	}
}
