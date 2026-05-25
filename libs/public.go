// @@
// @ Author       : Eacher
// @ Date         : 2026-05-23 09:42:34
// @ LastEditTime : 2026-05-25 17:30:32
// @ LastEditors  : Eacher
// @ --------------------------------------------------------------------------------<
// @ Description  : Please edit a descrition about this file at here.
// @ --------------------------------------------------------------------------------<
// @@
package libs

// #include "expand.h"
import "C"
import (
	"context"
	"errors"

	ggmlgo "ggml.go"
)

var is_init bool

func numa_init() {
	dev := C.ggml_backend_dev_by_type(C.GGML_BACKEND_DEVICE_TYPE_CPU)
	if dev != nil {
		panic("CPU backend is not loaded")
	}
	reg := C.ggml_backend_dev_backend_reg(dev)
	if reg != nil {
		panic("CPU backend is not loaded")
	}
	C.numa_init_fn(reg, C.GGML_NUMA_STRATEGY_NUMACTL)
}

type InitParams struct {
	Numa bool
}

func Init(p InitParams) error {
	if is_init {
		return errors.New("is init")
	}
	backend_init()
	if p.Numa {
		numa_init()
	}
	is_init = true
	return nil
}

func DInit() error {
	if is_init {
		backend_dinit()
		is_init = false
	}
	return nil
}

// file 模型绝对路径
func NewModel(file string, model *GGML) (GGML_INIT, error) {
	err := model.org.init(file, true)
	if err != nil {
		return nil, err
	}
	model.Alignment = uint64(C.gguf_get_alignment(model.org._fctx))
	model.MetaSize = uint64(C.gguf_get_meta_size(model.org._fctx))
	model.KV = int64(C.gguf_get_n_kv(model.org._fctx))
	model.Tensors = int64(C.gguf_get_n_tensors(model.org._fctx))
	return model.init, err
}

// file 模型绝对路径
func NewContext(file string, c *Context, ctx context.Context) error {
	err := c.org.init(file, false)
	if err == nil {
		c.org.ctx, c.org.cancel = context.WithCancelCause(ctx)
		go c.org.done()
	}
	return err
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

func LIB_ggml_blck_size(t ggmlgo.GGML_TYPE) int64 {
	// GGML_API int64_t ggml_blck_size(enum ggml_type type);
	return int64(C.ggml_blck_size(C.enum_ggml_type(t)))
}

func LIB_ggml_type_size(t ggmlgo.GGML_TYPE) uint64 {
	// GGML_API size_t  ggml_type_size(enum ggml_type type);             // size in bytes for all elements in a block
	return uint64(C.ggml_type_size(C.enum_ggml_type(t)))
}

func LIB_ggml_row_size(t ggmlgo.GGML_TYPE, ne int64) uint64 {
	// GGML_API size_t  ggml_row_size (enum ggml_type type, int64_t ne); // size in bytes for all elements in a row
	return uint64(C.ggml_row_size(C.enum_ggml_type(t), C.int64_t(ne)))
}

func LIB_ggml_ftype_to_ggml_type(t ggmlgo.GGML_FTYPE) ggmlgo.GGML_TYPE {
	// TODO: temporary until model loading of ggml examples is refactored
	// GGML_API enum ggml_type ggml_ftype_to_ggml_type(enum ggml_ftype ftype);
	return ggmlgo.GGML_TYPE(C.ggml_ftype_to_ggml_type(C.enum_ggml_ftype(t)))
}

func LIB_ggml_is_quantized(t ggmlgo.GGML_TYPE) bool {
	// GGML_API bool    ggml_is_quantized(enum ggml_type type);
	return bool(C.ggml_is_quantized(C.enum_ggml_type(t)))
}

func LIB_ggml_type_name(t ggmlgo.GGML_TYPE) string {
	// GGML_API const char * ggml_type_name(enum ggml_type type);
	return C.GoString(C.ggml_type_name(C.enum_ggml_type(t)))
}

func LIB_ggml_op_name(t ggmlgo.GGML_OP) string {
	// GGML_API const char * ggml_op_name  (enum ggml_op   op);
	return C.GoString(C.ggml_op_name(C.enum_ggml_op(t)))
}

func LIB_ggml_op_symbol(t ggmlgo.GGML_OP) string {
	// GGML_API const char * ggml_op_symbol(enum ggml_op   op);
	return C.GoString(C.ggml_op_symbol(C.enum_ggml_op(t)))
}

func LIB_ggml_unary_op_name(t ggmlgo.GGML_UNARY_OP) string {
	// GGML_API const char * ggml_unary_op_name(enum ggml_unary_op op);
	return C.GoString(C.ggml_unary_op_name(C.enum_ggml_unary_op(t)))
}

func LIB_ggml_glu_op_name(t ggmlgo.GGML_GLU_OP) string {
	// GGML_API const char * ggml_glu_op_name(enum ggml_glu_op op);
	return C.GoString(C.ggml_glu_op_name(C.enum_ggml_glu_op(t)))
}
