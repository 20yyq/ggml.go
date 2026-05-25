// @@
// @ Author       : Eacher
// @ Date         : 2026-05-23 09:05:18
// @ LastEditTime : 2026-05-24 11:24:31
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
	Model               string // 模型绝对路径
	KV, Tensors         int64  // KV 张量数
	Alignment, MetaSize uint64
	DataOffset          uint64 // 张量偏移
}

type Backend struct {
	org                 []backends
	Model               string // 模型绝对路径
	KV, Tensors         int64  // KV 张量数
	Alignment, MetaSize uint64
	DataOffset          uint64 // 张量偏移
}
