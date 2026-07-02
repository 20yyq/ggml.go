// @@
// @ Author       : Eacher
// @ Date         : 2026-05-23 09:05:18
// @ LastEditTime : 2026-06-30 10:38:35
// @ LastEditors  : Eacher
// @ --------------------------------------------------------------------------------<
// @ Description  : Please edit a descrition about this file at here.
// @ --------------------------------------------------------------------------------<
// @@
package libs

import (
	"context"
	"errors"
	"io"

	ggmlgo "ggml.go"
)

const GGML_MAX_DIMS = 4
const SIZE_MAX uint64 = 0xFFFFFFFFFFFFFFFF

type DevInfo struct {
	T                        ggmlgo.GGML_BACKEND_DEV_TYPE
	IsNuma                   bool
	DevName, DevDes, RegName string
	MemoryFree, MemoryTotal  uint64 // device free memory in bytes device total memory in bytes
}

type ResultTensor struct {
	Data []byte
	Info TensorInfo
}

type Dev struct {
	org  *dev
	idx  uint8
	Info DevInfo
}

type Backend struct {
	ctx               context.Context
	cancel            context.CancelCauseFunc
	is_init, is_close bool

	Dev Dev
}

func (ptr *Backend) Init(ctx context.Context) error {
	err := ptr.init()
	if err == nil {
		ptr.ctx, ptr.cancel = context.WithCancelCause(ctx)
		go ptr.done()
	}
	return err
}

func (ptr *Backend) Done() <-chan struct{} {
	if ptr.ctx == nil {
		return nil
	}
	return ptr.ctx.Done()
}

func (ptr *Backend) Set_n_threads(n uint16) error {
	if ptr.Dev.org == nil || !ptr.is_init || ptr.is_close {
		return errors.New("is close or is init")
	}
	return ptr.Dev.org.set_n_threads(ptr, n)
}

func (ptr *Backend) Close() error {
	if ptr.is_close {
		return errors.New("is close or is init")
	}
	if ptr.cancel != nil {
		ptr.cancel(io.EOF)
	}
	return nil
}

func (ptr *Backend) Check() error {
	if ptr.Dev.org == nil || !ptr.is_init || ptr.is_close {
		return errors.New("is close or is init")
	}
	return nil
}

func (org *Backend) init() error {
	err := errors.New("is close or is init")
	if org.Dev.org == nil || org.is_init || org.is_close {
		return err
	}

	if err = org.Dev.org.backend(org); err != nil {
		return err
	}
	org.is_init = true
	return nil
}

func (org *Backend) done() {
	<-org.ctx.Done()
	org.close()
}

func (org *Backend) close() error {
	err := errors.New("is close")
	if !org.is_close {
		org.is_close, err = true, nil
		org.Dev.org.delete_backend(org)
	}
	return err
}
