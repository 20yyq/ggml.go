// @@
// @ Author       : Eacher
// @ Date         : 2026-05-23 09:05:18
// @ LastEditTime : 2026-05-26 14:54:01
// @ LastEditors  : Eacher
// @ --------------------------------------------------------------------------------<
// @ Description  : Please edit a descrition about this file at here.
// @ --------------------------------------------------------------------------------<
// @@
package libs

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

type GGML struct {
	org                 ggml
	KV, Tensors         int64 // KV 张量数
	Alignment, MetaSize uint64
	DataOffset          uint64 // 张量偏移
}

type Context struct {
	org ggml
}

func (ptr *GGML) Close() error {
	err := ptr.org.close()
	if err == nil {
		ptr.org._fctx = nil
		ptr.org._gctx = nil
	}
	return err
}

func (ptr *Context) Close() error {
	err := ptr.org.close()
	if err == nil {
		ptr.org._fctx = nil
		ptr.org._gctx = nil
	}
	return err
}
