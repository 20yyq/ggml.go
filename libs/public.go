// @@
// @ Author       : Eacher
// @ Date         : 2026-05-23 09:42:34
// @ LastEditTime : 2026-05-23 22:27:47
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
	"context"
	"errors"
	"unsafe"
)

func NewIface(model *GGML, _ context.Context) (GGML_INIT, error) {
	err := error(nil)
	name, params := C.CString(model.Model), C.struct_gguf_init_params{no_alloc: true, ctx: &model.org._gctx}
	if model.org._fctx = C.gguf_init_from_file(name, params); model.org._fctx == nil {
		return nil, errors.New("failed to load model from gguf")
	}
	C.free(unsafe.Pointer(name))
	model.Alignment = uint64(C.gguf_get_alignment(model.org._fctx))
	model.MetaSize = uint64(C.gguf_get_meta_size(model.org._fctx))
	model.KV = int64(C.gguf_get_n_kv(model.org._fctx))
	model.Tensors = int64(C.gguf_get_n_tensors(model.org._fctx))
	return model.init, err
}

func LIB_ggml_version() string {
	return C.GoString(C.ggml_version())
}

func LIB_ggml_commit() string {
	return C.GoString(C.ggml_commit())
}

// use this to compute the memory overhead of a tensor
func LIB_ggml_tensor_overhead() uint64 {
	// GGML_API size_t ggml_tensor_overhead(void);
	return uint64(C.ggml_tensor_overhead())
}

func LIB_ggml_backend_dev_count() uint64 {
	// GGML_API size_t             ggml_backend_dev_count(void);
	return uint64(C.ggml_backend_dev_count())
}

func LIB_ggml_backend_reg_count() uint64 {
	// GGML_API size_t             ggml_backend_reg_count(void);
	return uint64(C.ggml_backend_reg_count())
}
