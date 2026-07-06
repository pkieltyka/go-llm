package providerutil

import llm "github.com/pkieltyka/go-llm"

// DerefPart normalizes a pointer part to its value form so adapter switches
// only handle value types (the llm.Part doctrine: parts are value types;
// pointer parts are normalized on entry). A typed nil pointer becomes an
// untyped nil Part. Extension and unknown parts pass through. This is a
// deliberate small duplication of the root package's private deref helper.
func DerefPart(part llm.Part) llm.Part {
	switch p := part.(type) {
	case *llm.TextPart:
		if p == nil {
			return nil
		}
		return *p
	case *llm.ImagePart:
		if p == nil {
			return nil
		}
		return *p
	case *llm.FilePart:
		if p == nil {
			return nil
		}
		return *p
	case *llm.ToolCallPart:
		if p == nil {
			return nil
		}
		return *p
	case *llm.ToolResultPart:
		if p == nil {
			return nil
		}
		return *p
	case *llm.ReasoningPart:
		if p == nil {
			return nil
		}
		return *p
	case *llm.UnknownPart:
		if p == nil {
			return nil
		}
		return *p
	default:
		return part
	}
}

// DerefEvent normalizes a pointer event to its value form so adapter
// switches only handle value types. A typed nil pointer becomes an untyped
// nil Event. Unknown events pass through.
func DerefEvent(event llm.Event) llm.Event {
	switch e := event.(type) {
	case *llm.MessageStart:
		if e == nil {
			return nil
		}
		return *e
	case *llm.TextDelta:
		if e == nil {
			return nil
		}
		return *e
	case *llm.ReasoningDelta:
		if e == nil {
			return nil
		}
		return *e
	case *llm.ToolCallStart:
		if e == nil {
			return nil
		}
		return *e
	case *llm.ToolCallDelta:
		if e == nil {
			return nil
		}
		return *e
	case *llm.ToolCallEnd:
		if e == nil {
			return nil
		}
		return *e
	case *llm.ToolCallDropped:
		if e == nil {
			return nil
		}
		return *e
	case *llm.MessageEnd:
		if e == nil {
			return nil
		}
		return *e
	default:
		return event
	}
}
