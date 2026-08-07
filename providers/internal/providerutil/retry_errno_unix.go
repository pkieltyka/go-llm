//go:build !windows && !plan9

package providerutil

import (
	"errors"
	"syscall"
)

var preSendDialErrors = []error{
	syscall.ECONNREFUSED,
	syscall.ENETUNREACH,
	syscall.EHOSTUNREACH,
	syscall.ETIMEDOUT,
}

func isConnectionReset(err error) bool {
	return errors.Is(err, syscall.ECONNRESET)
}
