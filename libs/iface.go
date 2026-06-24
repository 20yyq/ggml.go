// @@
// @ Author       : Eacher
// @ Date         : 2026-05-23 09:05:18
// @ LastEditTime : 2026-06-23 21:35:43
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
)

type GGML_IFACE interface {
}

type Model interface {
	Loader(GGML_IFACE)
	Devices(GGML_IFACE)
	LoadHparams(GGML_IFACE)
	LoadVocab(GGML_IFACE)
	LoadStatus(GGML_IFACE)
	LoadTensors(GGML_IFACE)
	// LoadHparams(GGML_IFACE)
}

type GGML_INIT func(Model) error

// type GGML struct {
// 	org                 ggml
// 	KV, Tensors         int64 // KV 张量数
// 	Alignment, MetaSize uint64
// 	DataOffset          uint64 // 张量偏移
// }

type Context struct {
	org GGML
}

func (ptr *GGML) Init(n uint64, cgraph bool, ctx context.Context) error {
	err := ptr.init(n, cgraph)
	if err == nil {
		ptr.ctx, ptr.cancel = context.WithCancelCause(ctx)
		go ptr.done()
	}
	return err
}

func (ptr *GGML) Close() error {
	if !ptr.is_init {
		return errors.New("is close or is init")
	}
	ptr.cancel(io.EOF)
	return nil
}

func (ptr *Context) Close() error {
	err := ptr.org.close()
	if err == nil {
		ptr.org._gctx = nil
	}
	return err
}
