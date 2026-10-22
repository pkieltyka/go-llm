package providerutil

import "time"

// StreamDrainGrace bounds the read past a stream's terminal event; a
// variable for tests.
var StreamDrainGrace = time.Second

// DrainStream reads a stream past its terminal event to EOF, so the
// connection can be reused and wire captures are complete. cancel (the
// stream's context) cuts off a server that holds the stream open longer
// than StreamDrainGrace.
func DrainStream(next func() bool, cancel func()) {
	timer := time.AfterFunc(StreamDrainGrace, cancel)
	defer timer.Stop()
	for next() {
	}
}
