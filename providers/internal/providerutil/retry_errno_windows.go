//go:build windows

package providerutil

import (
	"errors"
	"syscall"
)

// Windows network errors carry Winsock errno values rather than their POSIX
// counterparts. Keep both so synthetic tests and real dial failures classify
// consistently without adding a platform-specific dependency.
var preSendDialErrors = []error{
	syscall.Errno(10061), // WSAECONNREFUSED
	syscall.Errno(10051), // WSAENETUNREACH
	syscall.Errno(10065), // WSAEHOSTUNREACH
	syscall.Errno(10060), // WSAETIMEDOUT
}

func isConnectionReset(err error) bool {
	return errors.Is(err, syscall.Errno(10054)) // WSAECONNRESET
}
