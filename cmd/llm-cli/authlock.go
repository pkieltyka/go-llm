package main

import (
	"context"
	"fmt"
	"time"
)

const (
	maxAuthFileLockWait = 10 * time.Second
	authFileLockPoll    = 10 * time.Millisecond
)

type authFileLock interface {
	Release() error
}

func acquireAuthFileLock(ctx context.Context, authPath string) (authFileLock, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	waitCtx, cancel := context.WithTimeout(ctx, maxAuthFileLockWait)
	defer cancel()
	lockPath := authPath + ".lock"
	for {
		lock, acquired, err := tryAuthFileLock(lockPath)
		if err != nil {
			return nil, fmt.Errorf("lock auth file: %w", err)
		}
		if acquired {
			if err := waitCtx.Err(); err != nil {
				_ = lock.Release()
				return nil, err
			}
			return lock, nil
		}
		timer := time.NewTimer(authFileLockPoll)
		select {
		case <-timer.C:
		case <-waitCtx.Done():
			timer.Stop()
			return nil, waitCtx.Err()
		}
	}
}
