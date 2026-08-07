//go:build plan9

package providerutil

var preSendDialErrors []error

func isConnectionReset(error) bool { return false }
