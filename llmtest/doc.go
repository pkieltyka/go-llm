// Package llmtest provides a scripted fake llm.Provider for offline tests,
// plus the executable conformance suite for the llm.Provider contract.
//
// Like net/http/httptest, but for code that consumes go-llm: point the code
// under test at a Provider scripted with EnqueueResponse, EnqueueStream, and
// EnqueueError, then assert on the requests it recorded via Requests —
// hermetic tests with no network and no credentials. It is also the
// reference implementation of the llm.Provider contract.
//
// RunConformance is the checked form of the common lifecycle and normalized
// result contract: it verifies
// single-use and early-break stream semantics, prompt mid-stream context
// cancellation, successful MessageStart/MessageEnd grammar, empty/truncated
// EOF normalization, independent concurrent streams, panic-freedom on odd
// but valid requests, and Collect's partial-response-on-error shape. Every
// provider package in this module runs it against offline fixture servers,
// and third-party Provider implementations are encouraged to run it too.
//
// RunCapabilityConformance is the complementary native-activation suite for
// a reviewed subset of advertised capabilities. A CapabilityProfile supplies
// real requests and normalized assertions; an invocation-aware fixture owns
// exact provider wire assertions. CapabilityInvocation is fixture control data
// only and never changes a request or its model. Explicit exemptions record
// offline evidence gaps without denying the capability claim.
//
// Both suites are credential-free wire evidence. They do not prove live
// account/model availability, quota, cache admission, or service reliability;
// those belong in provider live tests.
//
// See examples/testing at the module root for a worked example of testing
// application code against this package.
package llmtest
