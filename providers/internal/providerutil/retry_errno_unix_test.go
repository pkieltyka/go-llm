//go:build !windows && !plan9

package providerutil

import (
	"net"
	"syscall"
	"testing"
)

func TestPreSendDialErrnosAndResetClassification(t *testing.T) {
	if !isProvablyPreSendError(&net.OpError{Op: "dial", Net: "tcp", Err: syscall.ECONNREFUSED}) {
		t.Fatal("refused dial was not classified as provably pre-send")
	}
	if isProvablyPreSendError(&net.OpError{Op: "read", Net: "tcp", Err: syscall.ECONNRESET}) {
		t.Fatal("connection reset was classified as provably pre-send")
	}
}
