// @@
// @ Author       : Eacher
// @ Date         : 2026-05-23 22:16:49
// @ LastEditTime : 2026-05-26 16:00:25
// @ LastEditors  : Eacher
// @ --------------------------------------------------------------------------------<
// @ Description  : Please edit a descrition about this file at here.
// @ --------------------------------------------------------------------------------<
// @@
package libs

// #include <stdlib.h>
// #include "ggml.h"
// #include "ggml-backend.h"
// #include "gguf.h"
import "C"
import (
	"fmt"

	ggmlgo "ggml.go"
)

func backend_init() {
	C.ggml_time_init()
	// needed to initialize f16 tables
	{
		var params C.struct_ggml_init_params
		var ctx *C.struct_ggml_context = C.ggml_init(params)
		C.ggml_free(ctx)
	}

	if C.ggml_backend_load_all(); C.ggml_backend_reg_count() == 0 {
		// hint: use ggml_backend_load() or ggml_backend_load_all() to load a backend before calling this function
		panic("no backends are loaded.")
	}
}

func backend_dinit() {
	C.ggml_quantize_free()
	fmt.Printf("backend_dinit \n")
}

type backend_t map[C.ggml_backend_dev_t]backend

func (ptr backend_t) find(dev C.ggml_backend_dev_t) backend {
	o, ok := ptr[dev]
	if !ok {
		ptr.update(dev, &o)
	}
	return o
}

func (ptr backend_t) update(dev C.ggml_backend_dev_t, o *backend) {
	var props C.struct_ggml_backend_dev_props
	C.ggml_backend_dev_get_props(dev, &props)
	o._dev, o.caps, o.memory_free, o.memory_total = dev, props.caps, uint64(props.memory_free), uint64(props.memory_total)
	o.T, o.Name, o.Des = ggmlgo.GGML_BACKEND_DEV_TYPE(props._type), C.GoString(props.name), C.GoString(props.description)
	ptr[dev] = *o
}

var maps = backend_t{}

type backend struct {
	_dev                      C.ggml_backend_dev_t
	caps                      C.struct_ggml_backend_dev_caps
	memory_free, memory_total uint64 // device free memory in bytes device total memory in bytes
	T                         ggmlgo.GGML_BACKEND_DEV_TYPE
	Name, Des                 string
}

type Backend struct {
	org         backend
	cur         C.ggml_backend_dev_t
	KV, Tensors int64 // KV 张量数
}
