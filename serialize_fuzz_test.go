package llm

import "testing"

func FuzzUnmarshalMessages(f *testing.F) {
	seeds := [][]byte{
		[]byte(`{"version":1,"messages":[]}`),
		[]byte(`{"version":1,"messages":[{"role":"user","parts":[{"type":"text","text":"hello","cache":{"ttl_ns":1}}]}]}`),
		[]byte(`{"version":1,"messages":[{"role":"user","parts":[{"type":"file","data":"ZmlsZQ==","media_type":"text/plain","name":"note.txt","cache":{"ttl_ns":5}}]}]}`),
		[]byte(`{"version":1,"messages":[{"role":"assistant","parts":[{"type":"reasoning","raw":{ "x": 1 },"provider":"openai"},{"type":"tool_call","id":"call_1","name":"lookup","args":{"q":"go"}}]}]}`),
		[]byte(`{"version":1,"messages":[{"role":"user","parts":[{"type":"future/part","payload":{ "html": "<tag>" }}]}]}`),
		[]byte(`{"version":2,"messages":[{"role":"user","parts":[{"text":"missing type"}]}]}`),
	}
	for _, seed := range seeds {
		f.Add(seed)
	}

	f.Fuzz(func(t *testing.T, data []byte) {
		msgs, err := UnmarshalMessages(data)
		if err != nil {
			return
		}
		out, err := MarshalMessages(msgs)
		if err != nil {
			t.Fatalf("MarshalMessages after successful decode returned error: %v", err)
		}
		if _, err := UnmarshalMessages(out); err != nil {
			t.Fatalf("UnmarshalMessages rejected remarshal %s: %v", out, err)
		}
	})
}
