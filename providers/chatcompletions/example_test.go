package chatcompletions_test

import (
	"os"

	"github.com/pkieltyka/go-llm/providers/chatcompletions"
)

// Example mirrors the README snippet: any OpenAI-compatible endpoint, with
// quirks declared as data.
func Example() {
	p, err := chatcompletions.New("https://api.example.com/v1",
		chatcompletions.WithName("example"),
		chatcompletions.WithAPIKey(os.Getenv("EXAMPLE_API_KEY")),
		chatcompletions.WithCompat(chatcompletions.Compat{StreamIncludeUsage: true}),
	)
	_ = p
	_ = err
	// Output:
}
