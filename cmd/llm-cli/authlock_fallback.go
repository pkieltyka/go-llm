//go:build aix || plan9 || js || wasip1

package main

import (
	"errors"
	"os"
)

type exclusiveAuthFileLock struct {
	file *os.File
	path string
}

func tryAuthFileLock(path string) (authFileLock, bool, error) {
	path += ".exclusive"
	file, err := os.OpenFile(path, os.O_CREATE|os.O_EXCL|os.O_RDWR, 0o600)
	if errors.Is(err, os.ErrExist) {
		return nil, false, nil
	}
	if err != nil {
		return nil, false, err
	}
	return &exclusiveAuthFileLock{file: file, path: path}, true, nil
}

func (lock *exclusiveAuthFileLock) Release() error {
	if lock == nil || lock.file == nil {
		return nil
	}
	closeErr := lock.file.Close()
	removeErr := os.Remove(lock.path)
	lock.file = nil
	if closeErr != nil {
		return closeErr
	}
	return removeErr
}
